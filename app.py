from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import os
import subprocess
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = r"C:\Users\IPASON\Desktop\API\reccorded"
FLUIDSYNTH_PATH = r"C:\Users\IPASON\Desktop\API\ffmpeg-7.1.1-essentials_build\bin\fluidsynth.exe"
FFMPEG_PATH = r"C:\Users\IPASON\Desktop\API\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
SOUNDFONT_PATH = r"C:\Users\IPASON\Desktop\API\soundfonts\FluidR3_GM.sf2"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

instrument_map = {
    'piano': ("Piano", 0),
    'flute': ("Flute", 73),
    'guitar': ("Guitar", 24),
    'violin': ("Violin", 40),
    'kalimba': ("Kalimba", 108),
    'trumpet': ("Trumpet", 56),
    'electric_guitar': ("Electric Guitar", 27),
    'cello': ("Cello", 42),
    'clarinet': ("Clarinet", 71),
    'tuba': ("Tuba", 58)
}

# AUDIO TO MIDI
def convert_audio_to_midi(audio_path, midi_path):
    print("Starting MIDI conversion...")
    model_output, midi_data, _ = predict(audio_path, ICASSP_2022_MODEL_PATH)
    midi_data.write(midi_path)
    print(f"MIDI saved to {midi_path}")
    return midi_path

# UPLOAD AUDIO FILE
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    m4a_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(m4a_path)

    wav_path = os.path.join(UPLOAD_FOLDER, "recorded.wav")
    midi_path = os.path.join(UPLOAD_FOLDER, "recorded_audio.mid")

    try:
        subprocess.run([FFMPEG_PATH, '-y', '-i', m4a_path, wav_path], check=True)
        convert_audio_to_midi(wav_path, midi_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'message': 'MIDI conversion complete',
        'midi_url': f"http://192.168.43.15:5000/converted/recorded_audio.mid"
    }), 200

@app.route('/converted/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# CONVERT MIDI TO INSTRUMENT AUDIO
@app.route('/convert_to_instrument', methods=['POST'])
def convert_to_instrument():
    data = request.get_json()
    instrument_key = data.get("instrument", "piano").lower()

    if instrument_key not in instrument_map:
        return jsonify({"error": "Invalid instrument name"}), 400

    instrument_name, instrument_number = instrument_map[instrument_key]
    midi_path = os.path.join(UPLOAD_FOLDER, "recorded_audio.mid")
    if not os.path.exists(midi_path):
        return jsonify({"error": "MIDI file not found"}), 404

    wav_path = os.path.join(UPLOAD_FOLDER, "temp.wav")
    adjusted_path = os.path.join(UPLOAD_FOLDER, "adjusted.wav")
    original_path = os.path.join(UPLOAD_FOLDER, "original.wav")

    midi = MidiFile(midi_path)
    new_midi = MidiFile()
    new_midi.ticks_per_beat = midi.ticks_per_beat

    for i, track in enumerate(midi.tracks):
        new_track = MidiTrack()
        used_channels = set()

        # Add tempo on the first track to ensure consistent timing
        if i == 0:
            new_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))

        for msg in track:
            if msg.type == 'program_change':
                continue
            if msg.type in ('note_on', 'note_off'):
                used_channels.add(msg.channel)
            new_track.append(msg)

        for ch in used_channels:
            if ch != 9:
                new_track.insert(1, Message('program_change', program=instrument_number, channel=ch, time=0))

        new_midi.tracks.append(new_track)

    new_midi.save(midi_path)

    try:
        subprocess.run(f'"{FLUIDSYNTH_PATH}" -ni -F "{wav_path}" "{SOUNDFONT_PATH}" "{midi_path}"', shell=True)

        # Load rendered audio
        y, sr = librosa.load(wav_path, sr=None)
        y = y / np.max(np.abs(y))  # Normalize

        # Save original for pitch shifting
        sf.write(original_path, y, sr)

        # Trim adjusted.wav to match original.wav duration
        original_duration = librosa.get_duration(filename=os.path.join(UPLOAD_FOLDER, "recorded.wav"))
        max_len = int(original_duration * sr)
        y_trimmed = y[:max_len]

        # Save trimmed version as adjusted.wav
        sf.write(adjusted_path, y_trimmed, sr)

    except Exception as e:
        return jsonify({"error": f"Audio rendering failed: {str(e)}"}), 500

    if os.path.exists(wav_path):
        os.remove(wav_path)

    return jsonify({
        "message": f"Instrument conversion to {instrument_name} complete.",
        "converted_file": f"http://192.168.43.15:5000/converted/adjusted.wav"
    }), 200

# ✅ FIXED: PITCH ADJUSTMENT WITH DURATION MATCHING
@app.route('/adjust_pitch', methods=['POST'])
def adjust_pitch():
    try:
        data = request.get_json()
        steps = int(data.get("steps", 0))

        original_path = os.path.join(UPLOAD_FOLDER, "original.wav")
        output_path = os.path.join(UPLOAD_FOLDER, "adjusted.wav")

        if not os.path.exists(original_path):
            return jsonify({"error": "original.wav not found"}), 404

        y, sr = librosa.load(original_path, sr=None)
        original_duration = librosa.get_duration(y=y, sr=sr)

        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        shifted_duration = librosa.get_duration(y=y_shifted, sr=sr)

        if abs(original_duration - shifted_duration) > 0.01:
            rate = original_duration / shifted_duration
            y_shifted = librosa.effects.time_stretch(y_shifted, rate)

        y_shifted = y_shifted / np.max(np.abs(y_shifted))
        sf.write(output_path, y_shifted, sr)

        return jsonify({
            "message": f"Pitch adjusted by {steps} steps.",
            "adjusted_file_url": f"http://192.168.43.15:5000/converted/adjusted.wav"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# CONCAT AUDIO FILES
@app.route('/concat', methods=['POST'])
def concat_audio():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        combined = None
        for f in files:
            filename = secure_filename(f.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(temp_path)
            audio = AudioSegment.from_file(temp_path)
            combined = audio if combined is None else combined + audio
            os.remove(temp_path)

        output_path = os.path.join(UPLOAD_FOLDER, 'merged.wav')
        combined.export(output_path, format='wav')

        return send_from_directory(app.config['UPLOAD_FOLDER'], 'merged.wav', as_attachment=True)

    except Exception as e:
        return jsonify({'error': f'Concat failed: {str(e)}'}), 500

# Start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
