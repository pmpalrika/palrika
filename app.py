from flask import Flask, request, jsonify
import pickle
from pipeline import process_data

app = Flask(__name__)

model = pickle.load(open('final.pkl', 'rb'))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    processed_input_data = process_data(data)

    prediction = model.predict(processed_input_data)

    output = prediction[0]

    return jsonify(output)


if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except Exception as e:
        print("Server is exited unexpectedly. Please contact server admin.")
        print(e)
