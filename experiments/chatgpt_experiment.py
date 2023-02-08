from chatgpt_wrapper import ChatGPT

prompt = """
Given the following context, what follows after this statement "# help me prepare egg and vegetable stir fry"? 

Context: 

available pick objects [apple, banana, eggplant, green beans, corn, carrot, kiwi, egg, strawberry, orange, yellow pepper, tomato sauce, cheese, milk]
available place_objects [soup pot, fry pan, pressure cooker, plate, electric stove, basket, sink, chopping board, blender]

# help me cook eggplant parmesan.
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(eggplant)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(tomato_sauce, fry pan)
robot.pick_and_place(cheese, fry pan)
done()

# help me prepare a fruit salad.
robot.pick_and_place(apple, chopping board)
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(kiwi, chopping board)
robot.pick_and_place(strawberry, chopping board)
robot.pick_and_place(orange, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(apple)
human.cut(banana)
human.cut(kiwi)
human.cut(strawberry)
human.cut(orange)
done()

# help me prepare tomato and vegetable omelette. 
robot.pick_and_place(tomato, chopping board)
robot.pick_and_place(eggplant, chopping board)
robot.pick_and_place(yellow_pepper, chopping board)
robot.pick_and_place(green_beans, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(tomato)
human.cut(eggplant)
human.cut(yellow_pepper)
human.cut(green_beans)
robot.pick_and_place(tomato, fry pan)
robot.pick_and_place(eggplant, fry pan)
robot.pick_and_place(yellow_pepper, fry pan)
robot.pick_and_place(green_beans, fry pan)
robot.pick_and_place(egg, fry pan)
done()

# help me prepare a banana and strawberry smoothie.
robot.pick_and_place(banana, chopping board)
robot.pick_and_place(strawberry, chopping board)
robot.pick_and_place(knife, chopping board)
human.cut(banana)
human.cut(strawberry)
robot.pick_and_place(banana, blender)
robot.pick_and_place(strawberry, blender)
robot.pick_and_place(milk, blender)
done()
"""

bot = ChatGPT()
response = bot.ask(prompt)
print(response)  # prints the response from chatGPT