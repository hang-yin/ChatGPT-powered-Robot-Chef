import openai
import time

"""
what are some simple meals I can cook with apple, banana, eggplant, 
green beans, corn, carrot, kiwi, egg, strawberry, orange, yellow pepper, 
tomato sauce, cheese and milk? 
I also have kitchenware such as soup pot, fry pan, pressure cooker, 
plate, electric stove, basket, sink, and chopping board. 
"""

"""
Menu:
    Fried Egg with Sautéed Vegetables: 
        Cook diced eggplant, green beans, and carrots in a frying pan with oil. 
        Serve with a fried egg and sliced yellow pepper on top.

    Tomato and Vegetable Omelette: 
        Mix diced eggplant, green beans, yellow pepper, 
        and chopped tomato in an egg mixture. 
        Cook as an omelette in a frying pan and top with grated cheese.

    Creamy Corn and Carrot Soup: 
        Cook diced carrots and corn in a soup pot with milk, 
        then puree until smooth. 
        Serve with diced yellow pepper and sliced kiwi on top.

    Baked Eggplant with Tomato Sauce and Cheese: 
        Slice eggplant and top with tomato sauce and grated cheese. 
        Bake in the oven until cheese is melted and eggplant is tender.

    Banana and Strawberry Smoothie: 
        Blend ripe banana and strawberries with milk until smooth. 
        Serve topped with sliced kiwi.

    Grilled Cheese and Orange Sandwich: 
        Toast bread in a frying pan with cheese and sliced orange between the slices.

    Egg and Vegetable Stir Fry: 
        Cook scrambled eggs and diced eggplant, green beans, 
        and yellow pepper in a frying pan with oil. Serve with sliced kiwi on top.
"""

openai_api_key = input("Enter your OpenAI API key: ")
openai.api_key = openai_api_key
# ENGINE = "text-davinci-001"
ENGINE = "text-ada-001"

PICK_TARGETS = {
  "apple": None,
  "banana": None,
  "eggplant": None,
  "green_beans": None,
  "corn": None,
  "carrot": None,
  "kiwi": None,
  "egg": None,
  "strawberry": None,
  "orange": None,
  "yellow_pepper": None,
  "knife": None,
  "tomato sauce": None,
  "cheese": None,
  "milk": None,
}

PLACE_TARGETS = {
  "soup_pot":            None,
  "fry_pan":             None,
  "pressure cooker":     None,
  "plate":               None,
  "electric stove":      None,
  "basket":              None,
  "sink":                None,
  "chopping board":      None,
  "blender":             None,
}

LLM_CACHE = {}

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    print('cache miss, calling')
    response = openai.Completion.create(engine=engine, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
    # time.sleep(30)
  return response

def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  response = gpt3_call(
      engine=engine, 
      prompt=gpt3_prompt_options, 
      max_tokens=0,
      logprobs=1, 
      temperature=0,
      echo=True,)
  
  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break
      total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)
    if options_in_api_form:
      option = "human.cut({})".format(pick)
      options.append(option)
  options.append(termination_string)
  print("Considering", len(options), "options")
  return options


termination_string = "done()"
gpt3_context = """
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
raw_input = "help me cook egg and vegetable stir fry." 
gpt3_prompt = gpt3_context + "\n#" + raw_input + "\n"
print(gpt3_prompt)
options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)

num_tasks = 0
max_tasks = 5
selected_task = ""
steps_text = []
all_llm_scores = []

while not selected_task == termination_string:
  num_tasks += 1
  if num_tasks > max_tasks:
    break

  llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
  selected_task = max(llm_scores, key=llm_scores.get)
  steps_text.append(selected_task)
  print(num_tasks, "Selecting: ", selected_task)
  gpt3_prompt += selected_task + "\n"

  all_llm_scores.append(llm_scores)

print('**** Solution ****')
print('# ' + raw_input)
for i, step in enumerate(steps_text):
  if step == '' or step == termination_string:
    break
  print('Step ' + str(i) + ': ' + step)
