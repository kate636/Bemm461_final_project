#Import packages
import pandas as pd
import plotly.express as px
import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
import ipywidgets as widgets
import plotly.graph_objs as go
import numpy as np
from IPython.display import display

# got tired of warnings 🙃
import warnings
warnings.filterwarnings('ignore')


jobs_all = pd.read_csv('gsearch_jobs.csv').replace("'","", regex=True)
jobs_all.date_time = pd.to_datetime(jobs_all.date_time) # convert to date time
jobs_all = jobs_all.drop(labels=['Unnamed: 0', 'index'], axis=1, errors='ignore')
jobs_all.description_tokens = jobs_all.description_tokens.str.strip("[]").str.split(",")

# Dictionary for skills and tools mapping, in order to have a correct naming
# Picked out keywords based on all keywords (only looked words with 100+ occurrences)
keywords_programming = [
'sql', 'python', 'r', 'c', 'c#', 'javascript', 'js',  'java', 'scala', 'sas', 'matlab', 
'c++', 'c/c++', 'perl', 'go', 'typescript', 'bash', 'html', 'css', 'php', 'powershell', 'rust', 
'kotlin', 'ruby',  'dart', 'assembly', 'swift', 'vba', 'lua', 'groovy', 'delphi', 'objective-c', 
'haskell', 'elixir', 'julia', 'clojure', 'solidity', 'lisp', 'f#', 'fortran', 'erlang', 'apl', 
'cobol', 'ocaml', 'crystal', 'javascript/typescript', 'golang', 'nosql', 'mongodb', 't-sql', 'no-sql',
'visual_basic', 'pascal', 'mongo', 'pl/sql',  'sass', 'vb.net', 'mssql', 
]

keywords_libraries = [
'scikit-learn', 'jupyter', 'theano', 'openCV', 'spark', 'nltk', 'mlpack', 'chainer', 'fann', 'shogun', 
'dlib', 'mxnet', 'node.js', 'vue', 'vue.js', 'keras', 'ember.js', 'jse/jee',
]

keywords_analyst_tools = [
'excel', 'tableau',  'word', 'powerpoint', 'looker', 'powerbi', 'outlook', 'azure', 'jira', 'twilio',  'snowflake', 
'shell', 'linux', 'sas', 'sharepoint', 'mysql', 'visio', 'git', 'mssql', 'powerpoints', 'postgresql', 'spreadsheets',
'seaborn', 'pandas', 'gdpr', 'spreadsheet', 'alteryx', 'github', 'postgres', 'ssis', 'numpy', 'power_bi', 'spss', 'ssrs', 
'microstrategy',  'cognos', 'dax', 'matplotlib', 'dplyr', 'tidyr', 'ggplot2', 'plotly', 'esquisse', 'rshiny', 'mlr',
'docker', 'linux', 'jira',  'hadoop', 'airflow', 'redis', 'graphql', 'sap', 'tensorflow', 'node', 'asp.net', 'unix',
'jquery', 'pyspark', 'pytorch', 'gitlab', 'selenium', 'splunk', 'bitbucket', 'qlik', 'terminal', 'atlassian', 'unix/linux',
'linux/unix', 'ubuntu', 'nuix', 'datarobot',
]

keywords_cloud_tools = [
'aws', 'azure', 'gcp', 'snowflake', 'redshift', 'bigquery', 'aurora',
]


# Define the custom function to extract the role
def get_role(title):
    roles = [
        "Data Engineer",
        "Data Scientist",
        "Data Analyst",
        "Business Analyst",
        "Software Engineer",
        "Cloud Engineer",
        "Machine Learning Engineer",
    ]
    if isinstance(title, str):
        for role in roles:
            if role.lower() in title.lower():
                return role
    return None

# Apply the custom function to create a new 'role' column
jobs_all["role"] = jobs_all["title"].apply(get_role)

# Define the custom function to extract the experience
def get_experience(title):
    if isinstance(title, str):
        title_lower = title.lower()
        if "lead" in title_lower:
            return "Lead"
        elif "senior" in title_lower:
            return "Senior"
        else:
            return "Junior"
    return None

# Apply the custom function to create a new 'experience' column
jobs_all["experience"] = jobs_all["title"].apply(get_experience)

# Define the custom function to extract the platform
def get_platform(via):
    platforms = [
        "Indeed",
        "LinkedIn",
        "Glassdoor",
        "Monster",
        "CareerBuilder",
        "SimplyHired",
        "Dice",
        "FlexJobs",
        "Snagajob",
        "ZipRecruiter",
        "LinkUp",
        "Idealist",
        "Craigslist",
        "USAJobs",
        "Google Jobs",
        "Robert Half",
        "AngelList",
        "The Ladders",
        "Guru",
        "Upwork",
        "Freelancer",
        "ArkLaMiss Jobs",
        "DiversityJobs",
    ]
    if isinstance(via, str):
        for platform in platforms:
            if platform.lower() in via.lower():
                return platform
    return None

# Apply the custom function to create a new 'platform' column
jobs_all["platform"] = jobs_all["via"].apply(get_platform)


# keyword_percentage_by_role_experience
def keyword_percentage_by_role_experience(data, roles, experiences, keyword_lists):
    # Combine all keywords into a single list
    all_keywords = []
    for keyword_list in keyword_lists:
        all_keywords += keyword_list

    # Initialize a dictionary to store the counts
    keyword_counts = {(role, experience): {keyword: 0 for keyword in all_keywords} for role in roles for experience in experiences}

    # Count the occurrences of each keyword in the job descriptions for each role and experience
    for index, row in data.iterrows():
        role = row["role"]
        experience = row["experience"]
        description = row["description"].lower()
        for keyword in all_keywords:
            if keyword.lower() in description:
                keyword_counts[(role, experience)][keyword] += 1

    # Calculate the percentage of each keyword for each role and experience
    keyword_percentages = {(role, experience): {} for role in roles for experience in experiences}
    for (role, experience), counts in keyword_counts.items():
        total_count = sum(counts.values())
        for keyword, count in counts.items():
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            keyword_percentages[(role, experience)][keyword] = percentage

    # Create a DataFrame with the results
    result_df = pd.DataFrame(keyword_percentages).stack(level=[0, 1]).reset_index()
    result_df.columns = ['keyword', 'role', 'experience', 'percentage']

    # Add keyword types column
    keyword_types = []
    for keyword in result_df['keyword']:
        if keyword in keywords_programming:
            keyword_types.append('programming')
        elif keyword in keywords_libraries:
            keyword_types.append('libraries')
        elif keyword in keywords_analyst_tools:
            keyword_types.append('analyst_tools')
        elif keyword in keywords_cloud_tools:
            keyword_types.append('cloud_tools')
        else:
            keyword_types.append('unknown')
    result_df['keywords_types'] = keyword_types

    return result_df

# Use the function on the given dataset and keyword lists
roles = jobs_all["role"].unique()
experiences = jobs_all["experience"].unique()
keyword_lists = [keywords_programming, keywords_libraries, keywords_analyst_tools, keywords_cloud_tools]
percentage_df = keyword_percentage_by_role_experience(jobs_all, roles, experiences, keyword_lists)




# Plot the interactive slope chart
def plot_interactive_slope_chart(data, role1, role2, role3, experience1, experience2, experience3, keywords_type, top_n=10):
    df = data.copy()
    
    filtered_df = df[((df.role == role1) | (df.role == role2) | (df.role == role3)) &
                     ((df.experience == experience1) | (df.experience == experience2) | (df.experience == experience3)) &
                     (df.keywords_types == keywords_type)]
    
    top_keywords = filtered_df.nlargest(top_n, 'percentage')['keyword'].unique()
    filtered_df = filtered_df[filtered_df.keyword.isin(top_keywords)]

    fig = go.Figure()

    for keyword in top_keywords:
        keyword_df = filtered_df[filtered_df.keyword == keyword]
        fig.add_trace(go.Scatter(x=keyword_df['role'] + '_' + keyword_df['experience'] + '_' + keyword_df['keywords_types'], 
                                 y=keyword_df.percentage,
                                 mode='lines+markers',
                                 name=keyword))
        
    fig.update_layout(title='Interactive Slope Chart',
                      xaxis_title=f'{role1}, {role2}, {role3}, {experience1}, {experience2}, {experience3}, {keywords_type}',
                      yaxis_title='Percentage')
    
    fig.show()

roles = percentage_df.role.unique()
experiences = percentage_df.experience.unique()
keywords_types = percentage_df.keywords_types.unique()

roles = [role for role in roles if role == role]
experiences = [experience for experience in experiences if experience == experience]
keywords_types = [keywords_type for keywords_type in keywords_types if keywords_type == keywords_type]

role1_widget = widgets.Dropdown(options=roles, description='Role 1:')
role2_widget = widgets.Dropdown(options=roles, description='Role 2:')
role3_widget = widgets.Dropdown(options=roles, description='Role 3:')

experience1_widget = widgets.Dropdown(options=experiences, description='Experience 1:')
experience2_widget = widgets.Dropdown(options=experiences, description='Experience 2:')
experience3_widget = widgets.Dropdown(options=experiences, description='Experience 3:')

keywords_type_widget = widgets.Dropdown(options=keywords_types, description='Keywords Type:')

role_widgets = widgets.HBox([role1_widget, role2_widget, role3_widget])
experience_widgets = widgets.HBox([experience1_widget, experience2_widget, experience3_widget])

ui = widgets.VBox([role_widgets, experience_widgets, keywords_type_widget])

out = widgets.interactive_output(plot_interactive_slope_chart,
                                 {'data': widgets.fixed(percentage_df),
                                  'role1': role1_widget,
                                  'role2': role2_widget,
                                  'role3': role3_widget,
                                  'experience1': experience1_widget,
                                  'experience2': experience2_widget,
                                  'experience3': experience3_widget,
                                  'keywords_type': keywords_type_widget,
                                  'top_n': widgets.fixed(10)})

display(ui, out)




"""
I apologize for the limited proficiency in Python visualization, despite my interest in the subject.
Initially, I had planned to utilize Dash to create a dashboard. 
However, during the course of my work, I encountered certain complexities in data processing, 
and creating a Slope chart consumed a significant amount of my time. 
Unfortunately, due to these unexpected challenges, I was unable to complete the dashboard in a timely manner.
"""
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Data Related Jobs Dashboard"),

#     dcc.Dropdown(
#         id="role-dropdown",
#         options=[{"label": role, "value": role} for role in jobs_all["role"].unique()],
#         value="Data Analyst",
#         clearable=False,
#         multi=True
#     ),

#     dcc.Graph(id="bar-chart")
# ])


# @app.callback(
#     Output("bar-chart", "figure"),
#     [Input("role-dropdown", "value")]
# )
# def update_bar_chart(selected_roles):
#     filtered_data = jobs_all[jobs_all["role"].isin(selected_roles)]

#     fig = px.bar(filtered_data, x="company_name", y="salary_avg",
#                  color="role", text="salary_avg",
#                  hover_data=["title", "location", "experience"],
#                  labels={"salary_avg": "Average Salary", "company_name": "Company Name"})

#     fig.update_layout(title="Average Salaries by Company",
#                       xaxis_title="Company",
#                       yaxis_title="Average Salary",
#                       legend_title="Role")
#     return fig


# if __name__ == "__main__":
#     app.run_server(debug=True, port = 8080)





















