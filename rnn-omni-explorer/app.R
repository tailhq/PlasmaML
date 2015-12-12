#! /usr/bin/Rscript
library(shiny)
library(markdown)
library(DT)
library(RSNNS)
library(ggplot2)
library(reshape2)
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
                     navbarMenu("Explore", 
                                tabPanel("Scatter Plot",
                                         sidebarLayout(
                                           sidebarPanel(
                                             selectInput("year1", "Year",
                                                         choices = c("2006" = "2006",
                                                                     "2007" = "2007"),
                                                         selected = "2006"),
                                             selectInput("xaxis", "X Axis",
                                                         choices = c("Solar Wind Speed" = "Vsw",
                                                                     "Interplanetary Magnetic Field Bz" = "Bz",
                                                                     "Proton Temperature" = "T",
                                                                     "Flow Pressure" = "P",
                                                                     "Sunspot Number" = "S"),
                                                         selected = "Vsw"),
                                             selectInput("yaxis", "Y Axis",
                                                         choices = c("Dst" = "Dst",
                                                                     "AE" = "AE",
                                                                     "Proton Temperature" = "T",
                                                                     "Flow Pressure" = "P"),
                                                         selected = "Dst")
                                             
                                           ),
                                           mainPanel(
                                             plotOutput("eplot")
                                           )
                                         )
                                ),
                                tabPanel("Correlation Heatmap",
                                         sidebarLayout(
                                           sidebarPanel(
                                             selectInput("year2", "Year",
                                                         choices = c("2006" = "2006",
                                                                     "2007" = "2007"),
                                                         selected = "2006"),
                                             selectInput("ctype", "Correlation Type",
                                                         choices = c("Pearson" = "pearson",
                                                                     "Spearman" = "spearman", 
                                                                     "Kendall" = "kendall"),
                                                         selected = "2006"),
                                             checkboxGroupInput("cvars", "Chosen Variables",
                                                                choices = c("Solar Wind Speed" = "Vsw",
                                                                            "Interplanetary Magnetic Field Bz" = "Bz",
                                                                            "Proton Temperature" = "T",
                                                                            "Flow Pressure" = "P",
                                                                            "Sunspot Number" = "S",
                                                                            "Dst" = "Dst",
                                                                            "AE" = "AE"
                                                                ),
                                                                selected = c("Vsw", "Bz", "Dst")
                                             )
                                           ),
                                           mainPanel(
                                             plotOutput("cplot")
                                           )
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
                   na.strings = c("99", "999.9",  "999", "99",
                                  "9999.", "9.999", "99.99", 
                                  "9999", "999999.99", 
                                  "99999.99", "9999999."))
    
    processedDF <- df[,c(17, 25, 41, 39, 40, 42, 23, 29)]
    
    colnames(processedDF) <- c("Bz", "Vsw", "Dst", "Kp", "S", "AE", "T", "P")
    #plot(processedDF$Vsw, processedDF$Dst, col = processedDF$Kp)
    finalDF <- processedDF[, c(input$xaxis, input$yaxis)]
    colnames(finalDF) <- c("input", "output")
    ggplot(finalDF, aes(x = input, y = output)) + geom_point()
    
  })
  
  output$cplot <- renderPlot({
    df <- read.csv(paste(prefix, input$year2, ".dat", sep = ""), 
                   header = FALSE, stringsAsFactors = FALSE, 
                   colClasses = rep("numeric",55), 
                   na.strings = c("99", "999.9",  "999", "99",
                                  "9999.", "9.999", "99.99", 
                                  "9999", "999999.99", 
                                  "99999.99", "9999999."))
    
    processedDF <- df[,c(17, 25, 41, 39, 40, 42, 23, 29)]
    
    colnames(processedDF) <- c("Bz", "Vsw", "Dst", "Kp", "S", "AE", "T", "P")
    cormat <- cor(processedDF[, input$cvars], use="complete.obs", method = input$ctype)
    melted_cormat <- melt(cormat)
    ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
      geom_tile()
    
  })
  
  output$table <- DT::renderDataTable({
    df <- read.csv(paste(prefix, "2006", ".dat", sep = ""), 
                   header = FALSE, stringsAsFactors = FALSE, 
                   colClasses = rep("numeric",55), 
                   na.strings = c("99", "999.9",  "999", "99",
                                  "9999.", "9.999", "99.99", 
                                  "9999", "999999.99", 
                                  "99999.99", "9999999."))
    
    processedDF <- df[,c(1,2,3,17, 25, 41, 39, 40, 42, 23, 29)]
    
    colnames(processedDF) <- c("Year","Day","Hour","Bz","Vsw", "Dst", "Kp", "S", "AE", "T", "P")
    DT::datatable(processedDF[,c("Year","Day","Hour","Vsw", "Bz", "Dst")])
  })
}

shinyApp(ui = ui, server = server)