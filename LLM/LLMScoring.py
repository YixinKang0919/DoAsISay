# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''

import openai


LLM_CACHE = {}


def gpt3_call(engine='text-ada-001', prompt='', max_tokens=128, temperature=0, logprobs=1, echo=False):
    full_query = ''
    for p in prompt: # prompt should be a list of str
        full_query += p
    _id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
    response = LLM_CACHE.get(_id)
    if response is None:
        response = openai.Completion.create(engine=engine, 
                                            prompt=prompt, 
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            logprobs=logprobs,
                                            echo=echo)
        LLM_CACHE[_id] = response
    return response
    
    # response = LLM_CACHE.get(_id, openai.Completion.create(engine=engine, 
    #                                                        prompt=prompt, 
    #                                                        max_tokens=max_tokens,
    #                                                        temperature=temperature,
    #                                                        logprobs=logprobs,
    #                                                        echo=echo))
    # LLM_CACHE[_id] = response
    # return response


def gpt3_scoring(query, options, engine='text-ada-001', limit_num_options: int=None, option_start='\n', verbose=False, print_tokens=False):
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and print(f'Scoring {len(options)} options')
    gpt3_prompt_options = [query + option for option in options]

    response = gpt3_call(engine=engine,
                         prompt=gpt3_prompt_options,
                         logprobs=1,
                         temperature=0,
                         echo=True)
    scores = {}
    for option, choice in zip(options, response['choices']):
        tokens = choice['logprobs']['tokens']
        token_logprobs = choice['logprobs']['token_logprobs']

        total_logprob = 0  # total token log prob for one option
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            print_tokens and print(token, token_logprob)
            if option_start is None and not token in option:
                break
            if token == option_start:
                break
            total_logprob += token_logprob
        scores[option] = total_logprob
    
    for i, option, in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        verbose and print(f'{option[1]}\t{option[0]}')
        if i > 10:
            break
    return scores, response


if __name__ == "__main__":
    d = {'a': -1, 'b': -2, 'c': 4}
    sd = sorted(d.items(), key=lambda x: -x[1]) # list, ascending order
    print(type(sd), sd)



    