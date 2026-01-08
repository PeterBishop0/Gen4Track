
from openai import OpenAI

_client = None  # global client

def openai_setup(key_path='./_OAI_KEY.txt'):
	global _client
	with open(key_path) as f:
		key = f.read().strip()

	print("Read key from", key_path)
	_client = OpenAI(api_key=key)

	print("[OpenAI] Client initialized with custom base_url")

def openai_completion(
	prompt,
	model='gpt-4o-mini',
	temperature=0,
	return_response=False,
	max_tokens=500,
	):
	
	assert _client is not None, "Call openai_setup() first"
	print("**prompt: ", prompt,'\n')
	resp =  _client.responses.create(
		model=model,
		input=prompt,
		# messages=[{"role": "user", "content": prompt}],
		temperature=temperature,
		max_output_tokens=max_tokens,
	)
	
	if return_response:
		return resp
	return  resp.output_text