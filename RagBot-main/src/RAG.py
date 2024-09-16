
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watson_machine_learning.foundation_models import Model
from langchain_core.documents import Document
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import json
import re

class RAG():
    #create a docstring for this whole class explaining what it does
    """This class implements the RAG model.

    The RAG model is a model that uses the Retriever, Augmenter, and Generator
    architecture to answer queries based on a given PDF file. The model is
    initialized with a PDF file, which is then split into pages and chunks.
    The model then initializes the prompts, retriever, and model. The model
    can then retrieve relevant documents based on a query, augment the prompt
    with the context and query, and generate an answer.

    Attributes:
    pages (list[Document]): The list of pages in the PDF file.
    all_splits (list[Document]): The list of chunks in the PDF file.
    recommend_course_prompt (str): The prompt for recommending a course.
    infer_context_prompt (str): The prompt for inferring the context.
    question_and_answer_prompt (str): The prompt for the questions and answers.
    retriever (Retriever): The retriever for retrieving relevant documents.
    model (Model): The model for generating answers.

    Methods:
    __init__: Initialize the RAG model with the given file name.
    __loadpdf__: Load the PDF file and split it into pages.
    __split__: Split the pages into chunks.
    __init_prompts__: Initialize the prompts.
    __init_retriever__: Initialize the retriever.
    __init_model__: Initialize the model.
    retrieve: Retrieve the relevant documents for the given query.
    augment: Augment the given template with the context and query.
    generate: Generate the answer for the given augmented prompt.
    answer_query: Generate the answer for the given context and query.

    """
    def __init__(self, file_name:str):
        """
        Initialize the RAG model with the given file name.

        Args:
        file_name (str): The name of the PDF file.
        """
        self.topics = ["AI", "DATA SCIENCE", "IBM AUTOMATION", "IBM CLOUD", "IBM ENGINEERING", "IBM SECURITY", "IBM Z", "IBM QUANTUM"]
        credentials = { 
            "url"    : "https://eu-gb.ml.cloud.ibm.com", 
            "apikey" : "MHmFq93k0rZxdsRQ6LYvXqqynpdA0nLbDw56-K_h7xt-"
        }
        self.__loadpdf__(file_name)
        self.__split__(self.pages)
        self.__init_prompts__()
        self.__init_retriever__()
        self.__init_model__(credentials)
    
    def __loadpdf__(self, file_name:str):
        """
        Load the PDF file and split it into pages.

        Args:
        file_name (str): The name of the PDF file.
        """
        self.pages = PyPDFLoader(file_name).load_and_split()

    def __split__(self, pages: list[Document]):
        """
        Split the pages into chunks.

        Args:
        pages (list[Document]): The list of pages in the PDF file.
        """
        # TODO: fix the split to work
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\nzblithereplz\n"],
            chunk_size=1500,
            chunk_overlap=250,
            length_function=len,
            is_separator_regex=False,
        )

        # new_pages = []
        # for page in pages:
        #     parts = page.page_content.split(":")
        #     for part in parts[1:]:
        #         separator_pos = part.find("\nzblithereplz\n")
        #         if separator_pos != -1:
        #             # Extract the text between the colon and the separator
        #             chunk = part[:separator_pos].strip()
        #             # Split this chunk into smaller chunks using RecursiveCharacterTextSplitter
        #             chunk_documents = text_splitter.split_documents([Document(chunk)])
        #             new_pages.extend(chunk_documents)
        # self.all_splits = new_pages
        self.all_splits = text_splitter.split_documents(pages)
    
    def __init_prompts__(self):
        """
        Initialize the prompt.
        """
        self.recommend_course_prompt = """
            Article:
            ###
            {}
            ###

            You are a chatbot that recommends courses to students. You are provided with an article that contains information about various courses.
            Based on the users input you will recommend 3 courses to the user.
            You can only recommend courses that are mentioned in the article.
            Answer in a complete sentence, with proper capitalization and punctuation. 
            Ommit verbose details from the article, but make sure to include the course name and link.
            If there is no good answer in the article, say "Sorry, I don't know".
            Do not link external resources.
            Always include the course name in your answer.
            Always include the course link in your answer.

            User input: {}
            1- Course and link: 
            2- Course and link:
            3- Course and link:
            """
        self.infer_context_prompt = """
            You are provided with a set of questions and answers.
            Your goal is to choose the most appropriate context based on the users answers to multiple question.
            You must choose the most appropriate context from the following list of contexts:
            1. AI
            2. DATA SCIENCE
            3. IBM AUTOMATION
            4. IBM CLOUD
            5. IBM ENGINEERING
            6. IBM SECURITY 
            7. IBM Z
            8. IBM QUANTUM
            Provide only the index of the context and no premable or explaination or text.
            Questions and Answers: 
            ###
            {}
            ###

            """
        self.question_and_answer_prompt = """
            Question: Can you describe any relevant work or study experience you have related to the courses you might want to try?
            Answer: {}
            Question: What is the highest level of education you have completed?
            Answer: {}
            Question: How would you rate your coding skills?
            Answer: {}
            Question: What career path are you aiming for, and how do you see these courses helping you achieve that?
            Answer: {}
            Question: Are there any particular technologies or skills you want to learn more about?
            Answer: {}
            """
        self.further_questions_prompt = """
            Article:
            ###
            {}
            ###

            You are a chatbot that recommends courses to students. You are provided with an article that contains information about various courses.
            Based on the users input you will recommend 3 courses to the user.
            You can only recommend courses that are mentioned in the article.
            Answer in a complete sentence, with proper capitalization and punctuation. 
            Only include the course name and link.
            Do not include any other information from the article.
            If there is no good answer in the article, say "Sorry, I don't know".
            Do not link external resources.
            Always include the course name in your answer.
            Always include the course link in your answer.
            Do not include the following courses:
            {}

            User input: {}
            1- Course and link: 
            2- Course and link:
            3- Course and link:
        """


    
    def __init_retriever__(self):
        """
        Initialize the retriever.
        """
        vectorstore = Chroma.from_documents(
            documents=self.all_splits,
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()
    
    def __init_model__(self, credentials: dict[str, str]):
        """
        Initialize the model.

        Args:
        credentials (dict[str, str]): The credentials for the model.
        """
        model_id = "mistralai/mixtral-8x7b-instruct-v01"

        gen_parms = { 
            GenParams.DECODING_METHOD : "greedy", 
            GenParams.MIN_NEW_TOKENS: 1, 
            GenParams.MAX_NEW_TOKENS: 500 
        }

        # project_id = os.environ["PROJECT_ID"]
        project_id = "6f8f36e6-af3d-4f52-afc2-5a7e34a0bda5"

        self.model = Model(model_id, credentials, gen_parms, project_id)

    def remove_course_descriptions(self, text: str) -> str:
        """
        Clean the output and remove the course descriptions from the text.

        Args:
        text (str): The text to remove the course descriptions from.

        Returns:
        str: The text with the course descriptions removed.
        """
        lines = text.splitlines()
        
        stripped_lines = [line.strip() for line in lines if line.strip()]
        
        stripped_text = "\n".join(stripped_lines)

        pattern = r"([^:]+:\s).*?(<https.+?>)"
        cleaned_text = re.sub(pattern, r"\1\2", stripped_text)
        return cleaned_text
    
    def infer_context(self, answers: dict) -> str:
        """
        Infer the context for the given answers.

        Args:
        answers (dict): The answers to the questions.
        """
        # infer context
        answers = answers.values()
        try:
            self.query = self.augment(self.question_and_answer_prompt, *answers)
        except IndexError:
            pass
        context_query = self.augment(self.infer_context_prompt, self.query)
        res = self.generate(context_query).replace(".", "")
        # print(res)
        try:
            context_index = int(res)
        except ValueError:
            context_index = 1
        return self.topics[context_index - 1]
    
    def retrieve(self, query: str) -> Document:
        """
        Retrieve the relevant documents for the given query, and remove unwanted characters.

        Args:
        query (str): The query to retrieve documents for.

        Returns:
        Document: The relevant document.
        """
        documents = self.retriever.get_relevant_documents(query)
        #TODO: implement a document picker
        articles = [documents[i].page_content for i in range(len(documents))]
        articles = [article.replace("\n", " ").replace("zblithereplz", "").replace("-", "").replace("—", "") for article in articles]
        return " ".join(articles)
    
    def augment(self, template_in: str, *args: tuple[str]) -> str:
        """
        Augment the given template with the context and query.

        Args:
        template_in (str): The template to augment.
        args (tuple[str]): A list of augments to augment the template with.

        Returns:
        str: The augmented template.
        """
        return template_in.format(*args)
    
    def generate(self, augmented_prompt_in: str) -> str:
        """
        Generate the answer for the given augmented prompt.

        Args:
        augmented_prompt_in (str): The augmented prompt to generate the answer for.

        Returns:
        str: The generated answer.
        """
        generated_response = self.model.generate( augmented_prompt_in )

        if ( "results" in generated_response ) \
        and ( len( generated_response["results"] ) > 0 ) \
        and ( "generated_text" in generated_response["results"][0] ):
            return generated_response["results"][0]["generated_text"]
        else:
            print( "The model failed to generate an answer" )
            print( "\nDebug info:\n" + json.dumps( generated_response, indent=3 ) )
            return ""
    
    def recommend_courses(self, answers: dict) -> dict:
        """
        Generate the answer for the given context and query.

        Args:
        answers (dict): The answers to the questions.
        """
        # infer context
        context = self.infer_context(answers)

        # retrieve documents
        documents = self.retrieve(context)

        # augment into prompt
        prompt = self.augment(self.recommend_course_prompt, documents, self.query)

        # generate answer
        generated_response = self.generate(prompt)

        # post-process the further questoins prompt for further recommendations
        self.further_questions_prompt = self.augment(self.further_questions_prompt, "{}", generated_response, self.query)

        # post-process the generated response
        generated_response = self.remove_course_descriptions(generated_response)

        # return response
        return {
                "text": generated_response,
                "results": [
                    {
                        "generated_text": generated_response[:100],
                        "generated_token_count": len(generated_response),
                        "input_token_count": len(prompt)
                    }
                ]
            }
    
    def recommend_other_courses(self, answers: dict) -> dict:
        """
        Generate further recommendation if initial were not satisfactory.

        Args:
        answers (dict): The answers to the questions.
        """
         # infer context
        context = self.infer_context(answers)

        # retrieve documents
        documents = self.retrieve(context)

        # augment into prompt
        self.further_questions_prompt = self.augment(self.further_questions_prompt, documents)

        # generate answer
        generated_response = self.generate(self.further_questions_prompt)

        # post-process the generated response
        generated_response = self.remove_course_descriptions(generated_response)
        return {
                "text": generated_response,
                "results": [
                    {
                        "generated_text": generated_response[:100],
                        "generated_token_count": len(generated_response),
                        "input_token_count": len(self.further_questions_prompt)
                    }
                ]
            }

    def recommend_known_topic(self, answers: dict) -> dict:
        """
        Generate the answer for the given context and query.

        Args:
        answers (dict): The answers to the questions.
        """
        # extract wanted course
        context = answers["ans1"]

        # retrieve documents
        documents = self.retrieve(context)

        # augment into prompt
        prompt = self.augment(self.recommend_course_prompt, documents, "Please recommend me a course.")

        # generate answer
        generated_response = self.generate(prompt)

        # post-process the generated response
        generated_response = self.remove_course_descriptions(generated_response)

        # return response
        return {
                "text": generated_response,
                "results": [
                    {
                        "generated_text": generated_response[:100],
                        "generated_token_count": len(generated_response),
                        "input_token_count": len(prompt)
                    }
                ]
            }
        