from flask import Flask, render_template, redirect, request, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from flask import send_file
import whisper, os, spacy, shutil, tempfile, zipfile

stopwords = list(STOP_WORDS)
punctuation = punctuation + '\n'

app = Flask(__name__)
app.secret_key = 'textsummary'

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///transcribe.db"
db = SQLAlchemy(app)
app.app_context().push()

class Transcription(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    text = db.Column(db.Text, nullable=False)
    sumText = db.Column(db.Text, nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.text}"


UPLOAD_FOLDER = r'D:\Flask\TextSummariser2\AudioFolder'
TRANSFORMED_FOLDER = r'D:\Flask\TextSummariser2\TransformedAudioFolder'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSFORMED_FOLDER'] = TRANSFORMED_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp3'}

def convert_audio(input_file, output_file):
    """
    Convert audio file to the recommended format for Whisper ASR.
    
    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the converted audio file.
    """
    audio = AudioSegment.from_mp3(input_file)
    audio = audio.set_frame_rate(16000)  # Set sampling rate to 16 kHz
    audio = audio.set_channels(1)  # Set number of channels to 1 (mono)
    audio = audio.set_sample_width(2)  # Set sample width to 16-bit (2 bytes)
    audio.export(output_file, format="wav")

@app.route('/', methods=['GET', 'POST'])
def file_upload(): 
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        print(file.filename)
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash("No selected file. Please select an MP3 file.", "error")
            return redirect(request.url)
        
        print(allowed_file(file.filename))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("File saved to:", file_path)
            
            # Convert audio file to recommended format for Whisper ASR
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_file_path = os.path.join(app.config['TRANSFORMED_FOLDER'], wav_filename)
            convert_audio(file_path, wav_file_path)
            print("Audio file converted to recommended format:", wav_file_path)

            model = whisper.load_model("base")
            print("Model loaded")

            transcription_text = model.transcribe(wav_file_path)
            print("Transcription Completed")
            transcription_result = transcription_text["text"]

            # Delete the uploaded MP3 file and the converted WAV file
            os.remove(file_path)
            os.remove(wav_file_path)
            
            #summarising the text using spaCy

            nlp = spacy.load('en_core_web_trf')
            doc = nlp(transcription_result)

            word_frequencies = {}
            for word in doc:
                if word.text.lower() not in stopwords:
                    if word.text.lower() not in punctuation:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] += 1

            max_frequency = max(word_frequencies.values())

            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word]/max_frequency

            sentence_tokens = [sent for sent in doc.sents]

            sentence_scores = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text.lower() in word_frequencies.keys():
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]

            from heapq import nlargest
            select_length = int(len(sentence_tokens)*0.3)
            select_length
            summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
            summary
            final_summary = [word.text for word in summary]
            summary = ' '.join(final_summary)

            transcription = Transcription(text=transcription_result, sumText = summary)
            db.session.add(transcription)
            db.session.commit()

            sno = transcription.sno
            return redirect(url_for('download_files', sno=sno)) 
        
        else:
            flash("Only MP3 files are allowed.", "error")
        
    return render_template('index.html')

@app.route('/download_files/<int:sno>')
def download_files(sno):

    transcription = Transcription.query.get_or_404(sno)

    # Create a temporary directory to store the text files
    temp_dir = tempfile.mkdtemp()

    # Write the transcribed text to a file
    text_filename = f"transcription.txt"
    text_filepath = os.path.join(temp_dir, text_filename)
    with open(text_filepath, 'w') as f:
        f.write(transcription.text)

    # Write the summarized text to a file
    summary_filename = f"summary.txt"
    summary_filepath = os.path.join(temp_dir, summary_filename)
    with open(summary_filepath, 'w') as f:
        f.write(transcription.sumText)

    # Create a zip file containing both text files
    zip_filename = f"transcription and summary.zip"
    zip_filepath = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        zipf.write(text_filepath, arcname=text_filename)
        zipf.write(summary_filepath, arcname=summary_filename)

    # Remove the records from the database
    db.session.delete(transcription)
    db.session.commit()

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    
    # Serve the zip file for download
    return send_file(zip_filepath, as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True)