# AlignAI MLOps Demo
The main demo for the MLOps course


## Running the container
### Dev
`docker build -f Dockerfile.dev -t alignai-mlops-demo .`
`docker run -p 8080:8080 -v --rm alignai-mlops-demo`

### Prod
`docker build -f Dockerfile -t alignai-mlops-demo-prod .`
`docker run -p 8080:8080 --rm alignai-mlops-demo-prod`

## Building and Deploying to Google App Engine

### Setup
1. [Read over the Google Tutorial on Deployment](https://cloud.google.com/appengine/docs/standard/python3/create-app)
2. [Create a new project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
3. [Go to the project selector](https://console.cloud.google.com/projectselector2/home/dashboard?_ga=2.80259487.928451217.1659464842-1605512166.1659464842&_gac=1.85381227.1659529353.Cj0KCQjwuaiXBhCCARIsAKZLt3nR_J-dUuSM9q2dhTtPPeOaU7Prn6CJeuDTuqfoj85Wq-fIKpKFxO4aAhWqEALw_wcB)
4. [Enable the API](https://console.cloud.google.com/flows/enableapi?apiid=cloudbuild.googleapis.com&_ga=2.80259487.928451217.1659464842-1605512166.1659464842&_gac=1.85381227.1659529353.Cj0KCQjwuaiXBhCCARIsAKZLt3nR_J-dUuSM9q2dhTtPPeOaU7Prn6CJeuDTuqfoj85Wq-fIKpKFxO4aAhWqEALw_wcB)
5. [Install](https://cloud.google.com/sdk/docs/install) and [initialize](https://cloud.google.com/sdk/docs/initializing) the Google Cloud CLI.
6. Initialize the app:
    `gcloud app create --project=[YOUR_PROJECT_ID]`

### Building with Docker
**Reference**: [Deploying Streamlit apps to Google App Engine in 5 simple steps.](https://medium.com/analytics-vidhya/deploying-streamlit-apps-to-google-app-engine-in-5-simple-steps-5e2e2bd5b172)