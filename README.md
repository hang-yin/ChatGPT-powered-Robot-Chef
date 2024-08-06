# Voice-controlled Robot Chef

This project is a system that enables voice-controlled, robot-assisted cooking. The system utilizes a custom Alexa skill to process user voice commands and a Flask app to generate recipe steps through ChatGPT. Object detection using a RealSense camera and the CLIP model allows the system to recognize objects in the kitchen and adjust recipe steps accordingly. The robot arm executes the steps autonomously, while a hand-action recognition model based on MediaPipe provides the necessary feedback to ensure collaboration between human and robot in completing the cooking tasks. This system contributes to home automation technology and showcases the integration of machine learning, computer vision, and robotics in real-world applications.

## Video Demo

https://user-images.githubusercontent.com/60046203/230218062-5e87c0ad-bf04-43af-933a-aebbcbe64365.mp4

## System Architecture

<img width="972" alt="winter-system" src="https://user-images.githubusercontent.com/60046203/230216902-7dffc861-c3d8-429c-9172-65d3f81e45cb.png">

## Alexa Custom Skill

To enable users to interact with this system more intuitively and conveniently, I built a custom Alexa skill that enables users to control the system through voice commands. The Alexa Developer console was used to build and deploy the custom skill on an Alexa dot, which converts audio to text and extracts the user's intent, which refers to the recipe they want to cook. Once the intent is extracted, the Alexa custom skill sends an HTTP request to a local Flask app, which would then process the request with ChatGPT to generate recipe steps that the robot arm can execute. 

## ChatGPT Query

### Overview

To process the user's intent and generate the recipe steps, the Flask app uses ChatGPT, a state-of-the-art language model that can process natural language input and generate text outputs. The Flask app builds up a prompt that includes example recipes, detected items using CLIP, and the current user intent/instruction. This prompt is then fed into ChatGPT, which generates a sequence of recipe steps that the robot arm can execute. By leveraging ChatGPT's advanced language processing capabilities, the system can provide users with detailed, context-sensitive instructions that take into account the available ingredients, the user's intent, and the current state of the kitchen. This allows the system to generate highly personalized and adaptive cooking instructions that are tailored to the specific needs of each user.

### Comparison with other GPT3 engines

During the development of the system, I evaluated several language models, including GPT-3 engines from OpenAI such as Ada 001, Davinci 001, and Davinci 003. We implemented a comparison test that utilized a simulated pick and place example code from the open-source [SayCan project](https://github.com/google-research/google-research/tree/master/saycan). The test involved building a collection of options based on the items available in the kitchen and then feeding them into the different language models to obtain scores via text completion. The highest-scoring action was then selected and added to the current context, with the process repeating until the recipe was complete. While the GPT-3 engines worked well for simple recipes, they struggled to handle more complicated recipes, particularly those that required human action. After comparing the results of each engine, we found that ChatGPT outperformed the others, particularly on more complex recipes. A comparison chart between all the engines and the result from ChatGPT is included below, highlighting the superiority of ChatGPT in handling complex recipe instructions.


| Language Engine | Recipe | Response |
|-----------------|--------|----------|
| Ada 001         | egg & vegetable stir fry | Step 0: robot.pick_and_place(egg, chopping board)<br />Step 1:humancut(egg)<br />Step 2: done()|
| Davinci 001     | egg & vegetable stir fry | Step 0: robot.pick_and_place(egg, chopping board)<br />Step 1: robot.pick_and_place(carrot, chopping board)<br />Step 2: robot.pick_and_place(yellow_pepper, chopping board)<br />Step 3: robot.pick_and_place(green_beans, chopping board)<br />Step 4: human.cut(egg)<br />Step 5: done()|
| Davinci 003     | egg & vegetable stir fry | Step 0: robot.pick_and_place(egg, chopping board)<br />Step 1: robot.pick_and_place(carrot, chopping board)<br />Step 2: robot.pick_and_place(green_beans, chopping board)<br />Step 3: robot.pick_and_place(yellow_pepper, chopping board)<br />Step 4: robot.pick_and_place(knife, chopping board)<br />Step 5: done()|
| ChatGPT         | egg & vegetable stir fry | Step 0: robot.pick_and_place(egg, fry pan)<br />Step 1: robot.pick_and_place(carrot, chopping board)<br />Step 2: robot.pick_and_place(corns, chopping board)<br />Step 3: robot.pick_and_place(green_beans, chopping board)<br />Step 4: robot.pick_and_place(knife, chopping board)<br />Step 5: human.cut(carrot)<br />Step 6: human.cut(corns)<br />Step 7: human.cut(green_beans)<br />Step 8: robot.pick_and_place(carrot, fry pan)<br />Step 9: robot.pick_and_place(corns, fry pan)<br />Step 10: robot.pick_and_place(green_beans, fry pan)<br />Step 11: done()|


### ChatGPT wrapper
It is worth noting that at the time of this project, the ChatGPT API was not yet released by OpenAI. Therefore, a [ChatGPT wrapper](https://github.com/mmabrouk/chatgpt-wrapper) was used to enable the integration of ChatGPT with the system. This wrapper allows the use of ChatGPT within a Python environment.

## Object Detection with CLIP

<img width="1105" alt="clip-result" src="https://user-images.githubusercontent.com/60046203/230217100-aafbd389-1ad7-43ad-a04d-c332118ca419.png">

### Comparison with traditional object detection methods
For object detection, the system uses a RealSense D435i camera and a machine learning model called CLIP. Unlike other state-of-the-art object detection models like DETR and YOLO v5, CLIP is trained on a massive dataset of 400 million (image, text) pairs, making it capable of recognizing out-of-vocabulary objects without the need for additional labeled data. Since CLIP can recognize objects based on their textual descriptions, it can also take into account different variations and synonyms of the same object, leading to a more robust detection performance. For example, we can ask CLIP to find a "red apple" or "chopped onion".  The 3D location of objects in the kitchen is provided by CLIP, which produces center object bounding boxes and uses depth information from the RealSense camera. Initially, I experimented with other object detection models but found that they couldn't recognize out-of-vocabulary objects without fine-tuning on every kitchen's novel objects, which was not practical. In contrast, CLIP's ability to recognize a wide range of objects without the need for additional labeling made it a more general and usable solution for the system.

### Implementation of CLIP for multiple objects

While the original CLIP model only performs image classification for a single object, we need to perform object detection for multiple objects in our kitchen. 

To utilize CLIP for object detection, I followed a tutorial called [Zero Shot Object Detection with OpenAI's CLIP](https://www.pinecone.io/learn/zero-shot-object-detection-clip/). The basic idea behind this approach is to break an image into many small patches, and pass a window over these patches, generating an image embedding for each unique window. We can then calculate the similarity between these patch image embeddings and our class label embeddings, returning a score for each patch. After calculating the similarity scores for every patch, we collate them into a map of relevance across the entire image. We use that map to identify the location of the object of interest. Here's an example for detecting an orange: 

<img width="1056" alt="clip-implementation" src="https://user-images.githubusercontent.com/60046203/230217217-b4631867-3553-4eeb-9794-408e3f171032.png">

Once we have identified the location of the object, we can recreate a bounding box around the object given a threshold for classification score. To achieve this, we used a RealSense D435i camera to obtain depth information, which we combined with the center of the object bounding box produced by CLIP to obtain 3D location information for the object.

## Hand Action Recognition with MediaPipe & LSTM

For more complex recipes that require human collaboration, our system needs to be able to recognize when the user has completed their assigned task. To address this, I developed a hand action recognition module using MediaPipe and an LSTM neural network. Since it's difficult for a single robot arm to perform tasks like cutting vegetables, we rely on the user to complete these tasks. However, we need a way to provide feedback to the system once the user has finished executing their tasks. To do this, I collected labeled video data and trained the LSTM neural network to classify video clips into hand actions such as grabbing and cutting behaviors. Instead of feeding entire video sequences into the neural network, I used MediaPipe to extract hand landmarks for faster processing and dimension reduction. Each video frame contained 21 hand landmarks with 3D coordinates. I collected 60 video clips for each hand action class, where each clip contained 45 frames. The LSTM was trained with categorical cross-entropy loss for 20 epochs and achieved 100% accuracy on the small testing set I collected. The architecture of the LSTM model is shown below:

<img width="820" alt="lstm-architecture" src="https://user-images.githubusercontent.com/60046203/230217324-c40c6413-3531-4dc4-bf52-b9c62f84f8bc.png">

The below video demonstrates the hand action recognition model in action.

https://user-images.githubusercontent.com/60046203/230218156-15704bfb-d69c-4ba7-8ed8-bcd6713d2da4.mp4

## Motion Module

The motion module of the system is built on top of a custom Python MoveIt API that my team and I built for controlling the Franka Emika Panda robot arm in a [previous project](https://hang-yin.github.io/portfolio/portfolio/jenga/). The API provides an interface to control the robot arm with a simple Python script by specifying Cartesian positions. In this current project, I used the same API to control the robot arm to perform the various actions required in the recipe. For example, given the position of an ingredient, the API can move the end effector of the robot arm to that position and perform the appropriate action, such as picking up or placing the ingredient. The API also provides collision detection and avoidance capabilities, ensuring the robot arm does not collide with any other objects in the workspace. By building on top of this existing API, I was able to rapidly prototype the motion module of the system and focus on the integration with the other modules.

## Usage Instruction
### Franka Emika Panda arm setup
- Plug into the Franka and the realsense camera.
- Log into station `ssh student@station`
- In the station, run `ros2 launch franka_moveit_config moveit.launch.py use_rviz:=false robot_ip:=panda0.robot`
- On your laptop, run `ros2 launch franka_moveit_config rviz.launch.py robot_ip:=panda0.robot`
- From the workspace containing these packages, run `ros2 run plan_execute main.launch.xml`

### Alexa and ChatGPT setup
- Run `chatgpt install` to make sure that your ChatGPT wrapper is working properly
- Run `ngrok http 5000` so that the Alexa custom skill can access our web server via the generated URL
- In Alexa developer console: 
  - Create a custom skill
  - Within this skill, create a user intent called "InstructionIntent" with a slot type called "food"
  - For this intent, some example utterences can be: "help me cook {food}", "help me prepare {food}"
  - Click `build model`
  - In the `code` section, copy and paste the file `lambda_function.py` in the `alexa` folder
  - The only thing you need to change in this file is the url in the `handle` method within the `InstructionIntentHandler` class; simply copy and paste the generated URL from ngrok
  - Then, click `Deploy`
- Once you finish setting up the Alexa custom skill, you can either:
  - Talk to your Alexa (signed in with the same Amazon account as your Alexa developer console)
    - `Alexa, open {robot kitchen assistant}` (you can change this by modifying the invocation name for your custom skill)
    - `Help me cook {some type of food}`
  - You can also open up the `Test` section on your Alexa developer console and type your voice commands 

## Reference
 - Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Ho, D., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jang, E., Ruano, R. J., Jeffrey, K., … Zeng, A. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv. [https://doi.org/10.48550/ARXIV.2204.01691](https://doi.org/10.48550/ARXIV.2204.01691)
 - Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv. [https://doi.org/10.48550/ARXIV.1906.08172](https://doi.org/10.48550/ARXIV.1906.08172)
 - Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv. [https://doi.org/10.48550/ARXIV.2103.00020](https://doi.org/10.48550/ARXIV.2103.00020)
