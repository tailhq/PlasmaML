#! /usr/bin/Rscript
library(shiny)
library(markdown)
library(DT)
library(RSNNS)
prefix <- "omni2_"
year <- 2006

layout <- navbarPage("Omni Explorer",
                     tabPanel("Test Models",
                              sidebarLayout(
                                sidebarPanel(
                                  selectInput("year", "Year",
                                              choices = c("2006" = "2006",
                                                          "2007" = "2007"),
                                              selected = "2006"
                                  ),
                                  checkboxGroupInput("netType", "Network Type",
                                                     choices = c("Jordan Network" = "j",
                                                                 "Elman Network" = "e"),
                                                     selected = "e"
                                  ), 
                                  sliderInput("trainingFraction",
                                              "Training Data Fraction",
                                              min = 0.01, max = 0.95, value = 0.75),
                                  sliderInput("learningParam",
                                              "Learning parameter: specifies the step width of the gradient descent",
                                              min = 0.001, max = 0.003, value = 0.005),
                                  sliderInput("maxGrowth",
                                              "Maximum growth parameter: specifies the maximum amount of weight change (relative to 1) which is added to the current change",
                                              min = 1.75, max = 2.25, value = 1.8),
                                  sliderInput("weightDecay",
                                              "Weight decay term to shrink the weights",
                                              min = 0.00001, max = 0.0009, value = 0.0001)
                                  
                                ),
                                mainPanel(
                                  plotOutput("plot")
                                )
                              )
                     ),
                     tabPanel("Explore",
                              sidebarLayout(
                                sidebarPanel(
                                  selectInput("year1", "Year",
                                              choices = c("2006" = "2006",
                                                          "2007" = "2007"),
                                              selected = "2006")
                                  
                                ),
                                mainPanel(
                                  plotOutput("eplot")
                                )
                              )
                     ),
                     navbarMenu("More",
                                tabPanel("Table",
                                         DT::dataTableOutput("table")
                                ),
                                tabPanel("About",
                                         fluidRow(
                                           column(6,
                                                  includeMarkdown("about.md")
                                           )
                                         )
                                )
                     )
)

ui <- fluidPage(layout)


server <- function(input, output, session) {
  
  output$plot <- renderPlot({
    
    df <- read.csv(paste(prefix, input$year, ".dat", sep = ""), 
                   header = FALSE, stringsAsFactors = FALSE, 
                   colClasses = rep("numeric",55), 
                   na.strings = c("99", "999.9", 
                                  "9999.", "9.999", "99.99", 
                                  "9999", "999999.99", 
                                  "99999.99", "9999999."))
    
    trainingLength <- as.integer(input$trainingFraction*nrow(df))
    trainFeatures <- df[1:trainingLength,c(17, 22, 25)]
    trainTargets <- df[1:trainingLength, 41]
    
    testFeatures <- df[(trainingLength + 1):nrow(df),c(17, 22, 25)]
    testTargets <- df[(trainingLength + 1):nrow(df),41]
    tr <- normalizeData(trainFeatures, type = "0_1")
    te <- normalizeData(testFeatures, attr(tr, "normParams"))
    trL <- normalizeData(trainTargets, type = "0_1")
    teL <- normalizeData(testTargets, attr(trL, "normParams"))
    
    colnames(tr) <- c("Bz", "SigmaBz", "Vsw")
    colnames(trL) <- c("Dst")
    
    colnames(te) <- c("Bz", "SigmaBz", "Vsw")
    colnames(teL) <- c("Dst")
    
    plot(testTargets, type = 'l', main = "Dst prediction", sub = as.character(year), 
         xlab = "Time (Hours)", ylab = "Dst")
    
    if("j" %in% input$netType) {
      modelJordan <- jordan(tr, trL, size = c(4),
                            learnFuncParams = c(input$learningParam, input$maxGrowth, input$weightDecay, 4),
                            maxit = 1000,
                            inputsTest = te,
                            targetsTest = teL,
                            linOut = TRUE, learnFunc = "QPTT")
      lines(denormalizeData(modelJordan$fittedTestValues, attr(trL, "normParams")), col="red")
      
    } 
    
    if("e" %in% input$netType) {
      modelEL <- elman(tr, trL, size = c(4),
                       learnFuncParams = c(input$learningParam, input$maxGrowth, input$weightDecay, 4),
                       maxit = 1000,
                       inputsTest = te,
                       targetsTest = teL,
                       linOut = TRUE, learnFunc = "QPTT")
      
      
      lines(denormalizeData(modelEL$fittedTestValues, attr(trL, "normParams")), col="green")
      
    }
    
  })
  
  output$eplot <- renderPlot({
    df <- read.csv(paste(prefix, input$year1, ".dat", sep = ""), 
                   header = FALSE, stringsAsFactors = FALSE, 
                   colClasses = rep("numeric",55), 
                   na.strings = c("99", "999.9", 
                                  "9999.", "9.999", "99.99", 
                                  "9999", "999999.99", 
                                  "99999.99", "9999999."))
    
    processedDF <- df[,c(17, 22, 25, 41)]
    
    colnames(processedDF) <- c("Bz", "SigmaBz", "Vsw", "Dst")
    plot(processedDF$Vsw, processedDF$Dst)
  })
  
  output$table <- DT::renderDataTable({
    DT::datatable(cars)
  })
}

shinyApp(ui = ui, server = server)