from dotenv import load_dotenv
from openai import OpenAI

class ChatBot:
    def __init__(self, model, system_message="You are a helpful assitant."):
        load_dotenv()
        self.client = OpenAI() # 클래스 내이기 때문에 self.client
        self.messages = []
        self.model = model
        self.add_message("system", system_message)

# add_message로 정의된 함수 내에서 role이랑 content가 상황에 따라 적절하게 바뀜
    def add_message(self, role, content):
        self.messages.append(
            {
                "role":role,
                "content": content
            }
        )

    def get_response(self, user_input, print_token_usage=False, response_format={"type":"text"}):
        self.add_message("user",user_input)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            response_format=response_format
        )

        if print_token_usage:
            print("="*50)
            # print(completion.usage.model_dump_json(indent=4))
            print("- 전송 토큰량:", completion.usage.prompt_tokens)
            print("- 응답 토큰량:", completion.usage.completion_tokens)
            print("- 전체 토큰량:", completion.usage.total_tokens)
            print("="*50)

        response = completion.choices[0].message.content
        self.add_message("assistant", response)
        return response
    
    # 토큰 정상화
    def reset(self):
        self.messages = self.messages[:1] # 시스템 메시지만 남기기. 첫번째 리스트 요소만 남기기