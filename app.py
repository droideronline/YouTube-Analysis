from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    jsonify,
)
from googleapiclient.discovery import build
from flask_cors import CORS
import re
from pytube import YouTube
from moviepy.editor import *
import os
import torch
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "ASR_model"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
API_KEY = os.getenv("API_KEY")
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe_ASR = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
output_path="./mp3_output/"
MODEL_ID="hateSpeech"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# setup pipeline as a text classification with multilabel outputs
pipe_hateSpeech = pipeline(
    task='text-classification',
    model=model,
    tokenizer=tokenizer,
    device=torch.cuda.current_device(),
    top_k=None
)

pipe_spam = pipeline("text-classification", model="spam_ham")
pipe_summarizer = pipeline("summarization", model="summarizer")
pipe_sentiment = pipeline("sentiment-analysis", model="sentiment", tokenizer="sentiment", return_all_scores=True)
chunk_size = 1024

# Function to chunk the text
def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def sanitize_filename(filename):
    # Remove invalid file name characters
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def counter(predictions):
    final = {label: 0 for label in ['disability', 'race', 'gender', 'origin', 'religion', 'age', 'sexuality']}
    for x in predictions:
        hate_speech_label = max(x, key=lambda x: x['score'])['label']
        final[hate_speech_label] += 1
    return final

def predictions_function(sentences):
    predictions = []
    for x in sentences:
        predictions.append(pipe_hateSpeech(x)[0])
    return predictions


@app.route('/convert', methods=['POST'])
def convert():
    # Get video
    content = request.json
    youtube_url = content['url']
    video = YouTube(youtube_url)
    print(youtube_url)
    stream = video.streams.get_highest_resolution()

    # Download video
    video_file_path = stream.download(skip_existing=True)
    
    # Load video using moviepy and extract audio
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    title = sanitize_filename(video.title)
    filename = output_path + title + ".mp3"
    audio_clip.write_audiofile(filename)

    # Free up memory
    audio_clip.close()
    video_clip.close()
    print("done until writing audio file")
    result = pipe_ASR(filename)
    print("ASR completed")
    # print("prediction =======",result["text"])

    if content['option'] == 'full' :
        sentences = result["text"]
    elif content['option'] == 'summarize':
        chunks = chunk_text(result['text'], chunk_size)
        summaries = [pipe_summarizer(chunk, max_length=130, min_length=30, do_sample=False) for chunk in chunks]
        sentences = ' '.join([summary[0]['summary_text'] for summary in summaries])
    print("sentences split")
    predictions = predictions_function(sentences)
    final = counter(predictions)
    print(predictions[0])
    print("hate speech predicted")
    # print(predictions)
    print(title + " has been successfully downloaded.")
    return jsonify({'message': 'File converted successfully', 'asr_content': result["text"], 'predictions':predictions, 'hate_speech':final})




@app.route('/analysis', methods=['POST', 'GET'])
def analysis():
    url = request.json['url']
    statistics = fetch_video_details(url)
    return statistics


def fetch_video_details(url):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_url = url
    
    # Extract video ID from URL
    video_id = video_url.split('v=')[1]
    
    # Fetch video details
    video_response = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    ).execute()
    
    video_details = video_response['items'][0]['snippet']
    video_statistics = video_response['items'][0]['statistics']
    
    # Fetch comments
    comments = []
    next_page_token = None
    while True:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=next_page_token
        ).execute()
        
        for item in comments_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        next_page_token = comments_response.get('nextPageToken')
        if not next_page_token:
            break

    # print(analyze_comments(comments))
    
    return jsonify({
        'title': video_details['title'],
        'description': video_details['description'],
        'likes': video_statistics.get('likeCount', 'Not available'),
        'comments': comments,
        'publishedAt': video_details['publishedAt'],
        'channel': video_details['channelTitle'],
        'tags': video_details.get('tags', []),
        'views': video_statistics.get('viewCount', 'Not available'),
        'commentCount': video_statistics.get('commentCount', 'Not available')
    })



@app.route("/hate_speech", methods=["POST","GET"])
def hate_speech():
    # print(pipe_hateSpeech(request.args.get('text')))
    return pipe_hateSpeech(request.args.get('text')[:512])
    

@app.route("/sentiment", methods=["POST","GET"])
def sentiment():
    # print(pipe_sentiment(request.args.get("text")))
    return pipe_sentiment(request.args.get("text")[:512])


@app.route("/summarize", methods=["POST","GET"])
def summarize():
    # print(pipe_summarizer(request.args.get('text'), max_length=200, min_length=70, do_sample=True))
    return pipe_summarizer(request.args.get('text'), max_length=200, min_length=70, do_sample=True)

@app.route("/spam_detection", methods=["POST","GET"])
def spam_detection():
    prediction = pipe_spam(request.args.get('text')[:512])
    # print(prediction)
    if prediction[0]['label']=='LABEL_0':
        return [[{'label':'Ham', 'score':prediction[0]['score']}]]
    elif prediction[0]['label']=='LABEL_1':
        return [[{'label':'Spam', 'score':prediction[0]['score']}]]



@app.route("/video", methods=["POST","GET"])
def forward():
    url = request.args.get('url', '')
    if url:
        url = 'https://www.youtube.com/watch?v=' + url
    return render_template('sample.html', url = url)




@app.route('/analysis-comments', methods=['POST'])
def analyze_comments():
    data = request.json
    comments = data['comments']
    # remove the below two lines if you want to get analysis for all the comments. present only 1000 comments are analyzed
    if len(comments) > 1002:
        comments = comments[:1001]
    # Initialize result storage
    sentiment_results = {'positive': 0, 'neutral': 0, 'negative': 0}
    spam_results = {'ham': 0, 'spam': 0}
    hate_speech_results = {label: 0 for label in ['disability', 'race', 'gender', 'origin', 'religion', 'age', 'sexuality']}
    
    for comment in comments:
        # Sentiment Analysis
        comment = comment[:512]
        sentiment = pipe_sentiment(comment)[0]
        sentiment_label = max(sentiment, key=lambda x: x['score'])['label']
        sentiment_results[sentiment_label] += 1
        
        # Spam Detection
        spam = pipe_spam(comment)[0]
        spam_label = 'ham' if spam['label'] == 'LABEL_0' else 'spam'
        spam_results[spam_label] += 1
        
        # Hate Speech Detection
        hate_speech = pipe_hateSpeech(comment)[0]
        hate_speech_label = max(hate_speech, key=lambda x: x['score'])['label']
        hate_speech_results[hate_speech_label] += 1

    # Aggregate and structure the results
    analysis_results = {
        'sentiment': sentiment_results,
        'spam': spam_results,
        'hate_speech': hate_speech_results
    }
    print(analysis_results)
    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
