import openai
import time

openai_api_key = ""
openai.api_key = openai_api_key
# ENGINE = "text-davinci-001"
ENGINE = "text-ada-001"

PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
}

"""
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,
"""

PLACE_TARGETS = {
  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
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
    time.sleep(30)
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

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options


termination_string = "done()"
gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""
raw_input = "put all the blocks in different corners." 
gpt3_prompt = gpt3_context + "\n#" + raw_input + "\n"
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
