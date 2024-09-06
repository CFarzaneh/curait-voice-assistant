import base64
from threading import Lock, Thread

import os
import json
import cv2
from langchain_openai import ChatOpenAI
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError
from macos_speech import Synthesizer
import llamaIndex

speaker = Synthesizer(voice='Samantha')

load_dotenv()


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)


        if "buy" and "amazon" in prompt.lower():
            print(prompt.lower())
            print("BUYING")
            vector_store = llamaIndex.initialize_index()
            query_engine = llamaIndex.get_query_engine(vector_store)
            response = llamaIndex.query("what is it that I want to "+prompt, query_engine)
            print(llamaIndex.amazon_purchase("Buy on amazon:" +response, query_engine))
        elif "recall" in prompt.lower():
            vector_store = llamaIndex.initialize_index()
            query_engine = llamaIndex.get_query_engine(vector_store)
            response = llamaIndex.query(prompt, query_engine)
            print("Response:", response)
            speaker.text = response
            speaker.talk()
        else:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image.decode()},
                config={"configurable": {"session_id": "unused"}},
            ).strip()
            speaker.text = response
            speaker.talk()

        if "remember this" in prompt.lower():
            self._save_response_to_json(prompt, response)

        if response:
            self._tts(response)

    def _save_response_to_json(self, prompt, response):
        output_dir = "output_json"
        output_file = os.path.join(output_dir, "response.json")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data = {
            "prompt": prompt,
            "response": response
        }

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
        else:
            existing_data = [data]

        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Response saved to {output_file}")

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
