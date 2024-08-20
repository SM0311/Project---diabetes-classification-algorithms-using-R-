library(shiny)
library(caret)
library(tidyverse)
library(rpart.plot)
library(MASS)
library(rsample)
library(dplyr)
library(randomForest)
library(plotly)
library(shinydashboard)
library(DT)
library(readr)
library(ranger)

# Load and preprocess the data
diabetes <- read.csv("diabetes.csv")

head(diabetes)
str(diabetes)
summary(diabetes)

# Split the data into training and testing sets
sampleData2 <- initial_split(diabetes, prop = 0.8, strata = Age)
trainset <- training(sampleData2)
testset <- testing(sampleData2)

# Convert Outcome category
trainset$Outcome <- factor(trainset$Outcome, labels = c("No", "Yes"))
testset$Outcome <- factor(testset$Outcome, labels = c("No", "Yes"))

# Train the models
# KNN Model
k_grid <- expand.grid(k = seq(1, 5))
KNN_model <- train(
  Outcome ~ .,
  data = trainset,
  method = "knn",
  preProcess = c("center", "scale"),
  tuneGrid = k_grid,
  trControl = trainControl(method = "cv", number = 10)
)

# Decision Tree Model
tree_diabetes <- train(
  Outcome ~ .,
  data = trainset,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10)
)

# Logistic Regression Model
Logistic_Diab <- train(
  Outcome ~ .,
  data = trainset,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

# Linear Discriminant Analysis Model
lda_model <- lda(Outcome ~ ., data = trainset)

# Random Forest Model
model_random <- train(
  Outcome ~ .,
  data = trainset,
  method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE, classProbs = TRUE),
  num.trees = 100,
  importance = "impurity"
)


# Define UI for the app
ui <- dashboardPage(
  dashboardHeader(title = "Diabetes Classification Models"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Prediction", tabName = "prediction", icon = icon("dashboard")),
      menuItem("Model Performance", tabName = "performance", icon = icon("chart-line"))
    )
  ),
  dashboardBody(
    tabItems(
      # Prediction tab
      tabItem(tabName = "prediction",
              fluidRow(
                box(title = "Input Features", status = "primary", solidHeader = TRUE,
                    numericInput("Pregnancies", "Number of Pregnancies:", value = 0, min = 0, max = 17),
                    numericInput("Glucose", "Glucose Level:", value = 0, min = 0, max = 199),
                    numericInput("BloodPressure", "Blood Pressure:", value = 0, min = 0, max = 122),
                    numericInput("SkinThickness", "Skin Thickness:", value = 0, min = 0, max = 99),
                    numericInput("Insulin", "Insulin Level:", value = 0, min = 0, max = 846),
                    numericInput("BMI", "Body Mass Index (BMI):", value = 0, min = 0, max = 67.10),
                    numericInput("DiabetesPedigreeFunction", "Diabetes Pedigree Function:", value = 0, min = 0.0780, max = 2.4200),
                    numericInput("Age", "Age:", value = 0, min = 21, max = 81),
                    actionButton("predict", "Predict", class = "btn-primary")
                ),
                box(title = "Feature Ranges", status = "info", solidHeader = TRUE,
                    p("Pregnancies: 0 - 17 "),
                    p("Glucose Level: 0 - 199"),
                    p("Blood Pressure: 0 - 122"),
                    p("Skin Thickness: 0 - 99"),
                    p("Insulin Level: 0 - 846"),
                    p("BMI: 0 - 67.10"),
                    p("Diabetes Pedigree Function: 0.0780 - 2.4200"),
                    p("Age: 21 - 81")
                ),
                box(title = "Model Predictions", status = "success", solidHeader = TRUE,
                    verbatimTextOutput("knnPrediction"),
                    verbatimTextOutput("treePrediction"),
                    verbatimTextOutput("logisticPrediction"),
                    verbatimTextOutput("ldaPrediction"),
                    verbatimTextOutput("rfPrediction")
                )
              )
      ),
      # Model Performance tab
      tabItem(tabName = "performance",
              fluidRow(
                box(title = "Model Performance Metrics", status = "warning", solidHeader = TRUE,
                    plotlyOutput("performancePlot")
                )
              )
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  observeEvent(input$predict, {
    # Create a new data frame with the input values
    new_data <- data.frame(
      Pregnancies = input$Pregnancies,
      Glucose = input$Glucose,
      BloodPressure = input$BloodPressure,
      SkinThickness = input$SkinThickness,
      Insulin = input$Insulin,
      BMI = input$BMI,
      DiabetesPedigreeFunction = input$DiabetesPedigreeFunction,
      Age = input$Age
    )
    
    # Predict using KNN
    knnPrediction <- predict(KNN_model, new_data)
    output$knnPrediction <- renderPrint({ paste("KNN Prediction: ", ifelse(knnPrediction == "Yes", "Patient has diabetes", "Patient does not have diabetes")) })
    
    # Predict using Decision Tree
    treePrediction <- predict(tree_diabetes, new_data)
    output$treePrediction <- renderPrint({ paste("Decision Tree Prediction: ", ifelse(treePrediction == "Yes", "Patient has diabetes", "Patient does not have diabetes")) })
    
    # Predict using Logistic Regression
    logisticPrediction <- predict(Logistic_Diab, new_data)
    output$logisticPrediction <- renderPrint({ paste("Logistic Regression Prediction: ", ifelse(logisticPrediction == "Yes", "Patient has diabetes", "Patient does not have diabetes")) })
    
    # Predict using LDA
    ldaPrediction <- predict(lda_model, new_data)
    output$ldaPrediction <- renderPrint({ paste("LDA Prediction: ", ifelse(ldaPrediction$class == "Yes", "Patient has diabetes", "Patient does not have diabetes")) })
    
    # Predict using Random Forest
    rfPrediction <- predict(model_random, new_data)
    output$rfPrediction <- renderPrint({ paste("Random Forest Prediction: ", ifelse(rfPrediction == "Yes", "Patient has diabetes", "Patient does not have diabetes")) })
  })
  
  # Example performance metrics plot (replace with actual performance metrics)
  output$performancePlot <- renderPlotly({
    # Example data for demonstration
    metrics <- data.frame(
      Model = c("KNN", "Decision Tree", "Logistic Regression", "LDA", "Random Forest"),
      Accuracy = c(0.85, 0.82, 0.88, 0.83, 0.90)
    )
    
    p <- ggplot(metrics, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity") +
      labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
      theme_minimal()
    
    ggplotly(p)
  })
}

# Run the application
shinyApp(ui = ui, server = server)