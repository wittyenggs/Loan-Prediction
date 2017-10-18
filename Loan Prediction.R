library('caret')
library('nnet')
library('e1071')

main <- function(){
  
  #Load the dataset
  data <- read.csv('IRIS.csv',header = TRUE, sep = ",")
  
  #Normalize the data
  data_norm <- predict(preProcess(data,method= c("range")),data)
  
  #Data Partition
  rows <- sample(nrow(data),floor(nrow(data)*0.70))
  train <- data_norm[rows,]
  test <- data_norm[-rows,]
  
  #Enlist Models
  models <- c('Logistic Regression', 'SVM')
  
  #Create Models, make predictions on test dataset and display confusion matrix
  for (model_name in models){
    model <- model(train,model_name)
    cm <- prediction(model,test)  
    print(paste("confusion matrix for", model_name)) 
    print (as.matrix(cm))
    
    
  }
}


model <- function (train,model_name){
  
  if (model_name == 'Logistic Regression')
    model <- multinom(Loan_Status ~., train)
  
  else if (model_name == 'SVM')
    model <- svm(Loan_Status ~ ., train)
  
  return(model)
  
}


prediction <- function(model,test){
  
  predictedValues <- predict(model,test, type = 'class')
  cm <- confusionMatrix(predictedValues,test$Loan_Status)
  return(cm)
}

main()