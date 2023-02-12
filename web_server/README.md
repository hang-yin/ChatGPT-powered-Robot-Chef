# Web Server Documentation

## Steps talk to Alexa
 - In this directory, run `python3 flask_ask_app.py` to run the flask app
 - Run `ngrok http 5000` to start ngrok, which would start a web tunnel at an HTTPS address
 - In the Alexa developer console, change the endpoint url address in the code panel to the ngrok HTTPS address
 - Start custom ALexa skill in the test panel by saying "open robot kitchen assistant"
