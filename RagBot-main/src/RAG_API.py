from RAG import RAG
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from datetime import datetime

rag = RAG("C:/Users/najla/Documents/CPC/College/Sheffield/Year 2/Semester 2/COM21002 AI Group Project/RagBot/resources/Knowlege_Base_for_LLM.pdf")
# answers = {"ans1": "I have completed a masterss degree", "ans2": "I rate my coding skills as advanced", "ans3": "I am aiming to become a software engineer, and I believe these courses will help me achieve that goal", "ans4": "I want to learn more about Cybersecurity", "ans5": "I am interested in learning about the security aspects of AI"}
# ans = rag.recommend_courses(answers)
# print(ans["text"])
# answers = {"ans1": "quantum computing", "ans2": "Beginner"}
# ans = rag.recommend_known_topic(answers)
# print(ans["text"])

app = Flask(__name__)
api = Api(app)

# argument parser for /recommend_known_topic
parser_recommend_known_topic = reqparse.RequestParser()
parser_recommend_known_topic.add_argument("ans1", type=str, required=True, help="Answer 1 is required.")
parser_recommend_known_topic.add_argument("ans2", type=str, required=True, help="Answer 2 is required.")


# argument parsing for /recommend
parser_recommend = reqparse.RequestParser()
parser_recommend.add_argument("ans1", type=str, required=True, help="Answer 1 is required.")
parser_recommend.add_argument("ans2", type=str, required=True, help="Answer 2 is required.")
parser_recommend.add_argument("ans3", type=str, required=True, help="Answer 3 is required.")
parser_recommend.add_argument("ans4", type=str, required=True, help="Answer 4 is required.")
parser_recommend.add_argument("ans5", type=str, required=True, help="Answer 5 is required.")

# argument parsing for /recommend_more
parser_recommend_more = reqparse.RequestParser()
parser_recommend_more.add_argument("ans1", type=str, required=True, help="Answer 1 is required.")
parser_recommend_more.add_argument("ans2", type=str, required=True, help="Answer 2 is required.")
parser_recommend_more.add_argument("ans3", type=str, required=True, help="Answer 3 is required.")

class RAG_API(Resource):
    def post(self):
        args = parser_recommend.parse_args()
        answers = {
            "ans1": args['ans1'],
            "ans2": args['ans2'],
            "ans3": args['ans3'],
            "ans4": args['ans4'],
            "ans5": args['ans5']
        }
        response_data = rag.recommend_courses(answers) 
        response = {
            "text": response_data['text'],
            "created_at": datetime.now().isoformat(),
            "results": response_data['results']
        }
        return jsonify(response)

class RAG_API_More(Resource):
    def post(self):
        args = parser_recommend_more.parse_args()
        answers = {
            "ans1": args['ans1'],
            "ans2": args['ans2'],
            "ans3": args['ans3']
        }
        response_data = rag.recommend_other_courses(answers) 
        response = {
            "text": response_data['text'],
            "created_at": datetime.now().isoformat(),
            "results": response_data['results']
        }
        return jsonify(response)

class RAG_API_Known_Topic(Resource):
    def post(self):
        args = parser_recommend_known_topic.parse_args()
        answers = {
            "ans1": args['ans1'],
            "ans2": args['ans2']
        }
        response_data = rag.recommend_known_topic(answers)
        response = {
            "text": response_data['text'],
            "created_at": datetime.now().isoformat(),
            "results": response_data['results']
        }
        return jsonify(response)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
api.add_resource(RAG_API, '/recommend')
api.add_resource(RAG_API_More, '/recommend_more')
api.add_resource(RAG_API_Known_Topic, '/recommend_known_topic')

if __name__ == "__main__":
    app.run(debug=True)

