from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool , tool
from langchain_groq import ChatGroq
import os , subprocess ,mdpdf

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768"
)

@tool
def convermarkdowntopdf(markdownfile_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using the mdpdf command line application.

    Args:
        markdownfile_name (str): Path to the input Markdown file.

    Returns:
        str: Path to the generated PDF file.
    """
    output_file = os.path.splitext(markdownfile_name)[0] + '.pdf'
    
    # Command to convert markdown to PDF using mdpdf
    cmd = ['mdpdf', '--output', output_file, markdownfile_name]
    
    # Execute the command
    subprocess.run(cmd, check=True)
    return output_file



markdown_to_pdf_creator = Agent(
    role='md to pdf Converter',
    goal='Convert the Markdown file to a PDF document. resume.md is the markdown file name.',
    backstory='An efficient converter that transforms Markdown files into professionally formatted PDF documents.',
    verbose=True,
    llm=llm,
    tools=[convermarkdowntopdf],
    allow_delegation=False
)

formatter = Agent(
    role = "Resume md formatter",
    goal = "The Formatters mission is to create a well-structured and readable “resume.md” file for the given user data. data : {data}",
    backstory='The Formatter is an expert in document formatting, having honed their skills through years of experience. They understand the importance of a well-organized resume and ensure that the content aligns with industry standards. Their backstory includes a passion for typography and an obsession with perfect alignment.',
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

enhancer = Agent(
    role='Enhancer',
    goal='The Enhancer aims to elevate the “resume.md” file, making it more professional and appealing to recruiters.',
    backstory='The Enhancer has a background in creative writing and design. Theyve worked with countless resumes and understand the psychology behind recruiter preferences. Their backstory involves a mentor who taught them the art of storytelling through bullet points',
    verbose=True,
    llm=llm,
    allow_delegation=False
)
task_format_content = Task(
    description='Format the given user info in markdown named resume.md . maintain the theme of resume. data: {data}',
    agent=formatter,
    expected_output='The entire resume content formatted in markdown, with important sections like experience,skills,details,links,etc according to user info . Use only information provided . do not use or add any external information. Return only the markdown content and nothing else',
    output_file="resume.md"
)

task_enhance_readme = Task(
    description='Enhance the README content for a project',
    agent=enhancer,
    context = [task_format_content],
    expected_output='The improved README content with more professional content, better formatting, and relevant sections to attract recruiters. Return only the markdown content and nothing else',
    output_file='resume.md'
)
convert = Task(
     description='convert the markdown named resume.md into resume.pdf without change in its data.  maintain the theme of resume.',
    agent=markdown_to_pdf_creator,
    context = [task_enhance_readme],
    expected_output='The entire resume content formatted in markdown, with important sections like experience,skills,details,links,etc according to user info . Use only information provided . do not use or add any external information. Return only the markdown content and nothing else',
    output_file="resume.pdf"
)
crew = Crew(
    tasks=[task_format_content,task_enhance_readme,convert],
    agents=[formatter,enhancer,markdown_to_pdf_creator],
    process=Process.sequential,
    verbose=True,
)

user_data = {
    "full_name": "Yadv ji",
    "email": "john.doe@email.com",
    "phone": "(123) 456-7890",
    "linkedin": "https://www.linkedin.com/in/johndoe",
    "professional_summary": "A highly motivated data entry professional with a keen eye for detail and accuracy. Proficient in managing large volumes of data and ensuring data integrity. Seeking opportunities to contribute my skills in a dynamic work environment.",
    "work_experience_position1": "Data Entry Specialist",
    "work_experience_company1": "XYZ Healthcare",
    "work_experience_duration1": "Jan 2022 - Present",
    "work_experience_responsibilities1": [
        "Managed patient records, ensuring accuracy and compliance with privacy regulations.",
        "Entered medical data into electronic health records (EHR) system.",
        "Collaborated with healthcare professionals to maintain up-to-date records."
    ],
    "work_experience_position2": "Data Entry Intern",
    "work_experience_company2": "ABC Manufacturing",
    "work_experience_duration2": "Summer 2021",
    "work_experience_responsibilities2": [
        "Assisted in digitizing inventory records and updating databases.",
        "Conducted data validation and resolved discrepancies.",
        "Improved data entry efficiency by implementing streamlined processes."
    ],
    "skills": [
        "Data Entry",
        "Microsoft Excel",
        "Attention to Detail",
        "Time Management",
        "Communication"
    ],
    "education_degree": "Bachelor of Science in Business Administration",
    "education_university": "University of Data Management",
    "education_graduation_date": "May 2021",
    "certifications": [
        "Certified Data Entry Specialist (CDES)"
    ]
}

user_data_string = str(user_data)

result = crew.kickoff(inputs={"data" : user_data_string})
print(result)