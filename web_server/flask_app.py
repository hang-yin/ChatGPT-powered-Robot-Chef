from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/alexa", methods=["POST"])
def alexa():
    print("Received request from Alexa")
    request_data = request.get_json()
    # Process the request from Alexa
    # ...
    print("Processed request from Alexa")
    # Return a response to Alexa
    response = {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": "Hello from Flask!"
            },
            "shouldEndSession": True
        }
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

