import openai

# openai_api_key = ""
# openai.api_key = openai_api_key
# ENGINE = "text-davinci-001"
# ENGINE = "text-ada-001"

LLM_CACHE = {}

class GPT_Solver():
    """
    TODO: add docstring
    """

    def __init__(self, engine="text-ada-001", termination_string="done()"):
        self.engine = engine
        self.pick_targets = {
            "apple": None,
            "banana": None,
            "eggplant": None,
            "green beans": None,
        }
        self.place_targets = {
            "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
            "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
            "middle":              (0,           -0.5,        0),
            "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
            "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
        }
        self.gpt3_context = """
        # move all fruits and vegetables to the top left corner.
        robot.pick_and_place(apple, top left corner)
        robot.pick_and_place(eggplant, top left corner)
        robot.pick_and_place(banana, top left corner)
        robot.pick_and_place(green beans, top left corner)
        done()

        # put the fruits in the top right corner.
        robot.pick_and_place(apple, top right corner)
        robot.pick_and_place(banana, top right corner)
        done()

        # move the vegetables to the middle.
        robot.pick_and_place(eggplant, middle)
        robot.pick_and_place(green beans, middle)
        done()
        """
        self.max_tasks = 5
        self.termination_string = termination_string
    
    def make_options(self):
        options = []
        for pick in self.pick_targets:
            for place in self.place_targets:
                option = "robot.pick_and_place({}, {})".format(pick, place)
                options.append(option)
        options.append(self.termination_string)
        print("Considering", len(options), "options")
        return options

    def gpt3_call(self, prompt, max_tokens=128, logprobs=1, temperature=0, echo=True):
        """
        full_query = ""
        for p in prompt:
            full_query += p
        id = tuple((self.engine, full_query, max_tokens, temperature, logprobs, echo))
        if id in LLM_CACHE.keys():
            print('cache hit, returning')
            response = LLM_CACHE[id]
        """
        # else:
        # print('cache miss, calling')
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
        )
        # LLM_CACHE[id] = response
        return response
    
    def gpt3_scoring(self, query, options, option_start="\n"):
        gpt3_prompt_options = [query + option for option in options]
        response = self.gpt3_call(
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
                if option_start is None and not token in option:
                    break
                if token == option_start:
                    break
                total_logprob += token_logprob
            scores[option] = total_logprob
        for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
            print(option[1], "\t", option[0])
            if i >= 10:
                break
        return scores
    
    def solve(self, raw_input):
        gpt3_prompt = self.gpt3_context + "\n#" + raw_input + "\n"
        num_tasks = 0
        steps_text = []
        selected_task = ""
        options = self.make_options()
        
        while not selected_task == self.termination_string:
            num_tasks += 1
            if num_tasks > self.max_tasks:
                break
            llm_scores = self.gpt3_scoring(gpt3_prompt, options)
            selected_task = max(llm_scores, key=llm_scores.get)
            steps_text.append(selected_task)
            print(num_tasks, "Selecting: ", selected_task)
            gpt3_prompt += selected_task + "\n"
        
        self.print_solution(raw_input, steps_text)
        return steps_text
    
    def print_solution(self, raw_input, steps_text):
        print('**** Solution ****')
        print('# ' + raw_input)
        for i, step in enumerate(steps_text):
            if step == '' or step == self.termination_string:
                break
            print('Step ' + str(i) + ': ' + step)

if __name__ == "__main__":
    # ask for user input: api key
    openai_api_key = input("Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    solver = GPT_Solver()
    _ = solver.solve("put all the fruits in the middle.")
    