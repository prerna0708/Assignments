#AUS 
rm(list=ls())
library(dplyr)
set.seed(100)
######################UNIVARIATE##########################
#read in data#

# Import data set
GDP = read.csv(file = file.choose(),header = T)
unemployment = read.csv(file = file.choose(),header = T)

unemployment_AUSgium = unemployment %>% filter(ï..LOCATION == "AUS") %>% select(TIME,Value)%>% filter(TIME>1969 & TIME<2020)%>% rename(Year = TIME, unemployment_Value = Value) 

library(dplyr)
GDP_AUSgium = GDP %>% filter(ï..LOCATION == "AUS") %>%  
  select(TIME,Value) %>% filter(TIME>1969 & TIME<2020) %>% 
  rename(Year = TIME, GDP_Value = Value)

attach(GDP_AUSgium)

library(CADFtest)
library(xtable)
library(vars)

GDP_ts <- ts (GDP_Value, frequency = 1, start = c(1970,1))
#log#
log_GDP_ts<- log (GDP_ts)
plot.ts(log_GDP_ts)
plot.ts(GDP_ts)

#plotting the time series (FIGURE 1)
plot.ts(log_GDP_ts, ylab="GDP", main="Total GDP per year")
#unit root test for stationarity to confirm observation
max.lag<-round(sqrt(length(log_GDP_ts)))
CADFtest(log_GDP_ts, type ="trend", criterion="BIC", max.lag.y = max.lag) #p-value>0.05

#Go in differences# to remove the trend
dlogGDP_ts <- diff(log_GDP_ts)
plot.ts(dlogGDP_ts, ylab="Change in GDP", main="Change in GDP")
acf(dlogGDP_ts)#MA(2)
pacf(dlogGDP_ts)#AR(0)

#unit root test for stationarity to confirm observation
max.lag<-round(sqrt(length(dlogGDP_ts)))
CADFtest(dlogGDP_ts, type ="drift", criterion="BIC", max.lag.y = max.lag) #p-value<0.05 reject H0#

###################################ARIMA MODELLING#######################################
#First ARIMA Model#
fit1 <- arima(log_GDP_ts, order=c(0,1,2))
fit1
summary(fit1)
#AIC=-158.11, BIC=-152.43
BIC(fit1)
abs(fit1$coef/sqrt(diag(fit1$var.coef)))
(1-pnorm(abs(fit1$coef/sqrt(diag(fit1$var.coef)))))*2
#significant terms:MA1, MA2
plot.ts(fit1$residuals)
acf(fit1$residuals, lag.max=max.lag)
pacf(fit1$residuals, lag.max=max.lag)

Box.test(fit1$residuals, lag = 1, type = "Ljung-Box") 
#p-value <0.05##Do Reject H0## not white noise#not valid model

fit12 <- arima(log_GDP_ts, order=c(1,1,1))
fit12
summary(fit12)
#AIC= -206.13, BIC=-200.4
BIC(fit12)
abs(fit12$coef/sqrt(diag(fit12$var.coef)))
(1-pnorm(abs(fit12$coef/sqrt(diag(fit12$var.coef)))))*2
#No term is significant
plot.ts(fit12$residuals)
acf(fit12$residuals, lag.max=max.lag)
pacf(fit12$residuals, lag.max=max.lag)

Box.test(fit12$residuals, lag = 1, type = "Ljung-Box") 
#p-value of 0.348##do not Reject H0## white noise#valid model

fit14 <- arima(log_GDP_ts, order=c(1,1,2))
fit14
BIC(fit14)
summary(fit14)
#AIC= -205.19, -197.62
abs(fit14$coef/sqrt(diag(fit14$var.coef)))
(1-pnorm(abs(fit14$coef/sqrt(diag(fit14$var.coef)))))*2
#ar1, ma1 significant
plot.ts(fit14$residuals)
acf(fit14$residuals, lag.max=max.lag)
pacf(fit14$residuals, lag.max=max.lag)

Box.test(fit14$residuals, lag = 1, type = "Ljung-Box") 
#p-value of 0.99

fit15 <- arima(log_GDP_ts, order=c(1,1,3))
fit15
BIC(fit15)
summary(fit15)
#AIC= -204.32
abs(fit15$coef/sqrt(diag(fit15$var.coef)))
(1-pnorm(abs(fit15$coef/sqrt(diag(fit15$var.coef)))))*2

plot.ts(fit15$residuals)
acf(fit15$residuals, lag.max=max.lag)
pacf(fit15$residuals, lag.max=max.lag)

Box.test(fit15$residuals, lag = 1, type = "Ljung-Box") 
#p-value of 0.85 #VALID MODEL


#forecast 
myforecast<-predict(fit14, n.ahead=8)
expected<-myforecast$pred
lower<-myforecast$pred-qnorm(0.975)*myforecast$se
upper<-myforecast$pred+qnorm(0.975)*myforecast$se
plot.ts(log_GDP_ts, xlim=c(2008,2028), ylim=c(10.5,20.5))
lines(expected, col="red")
lines(lower, col="blue")
lines(upper, col="blue")
myforecast$se
###prediction with the best model 


y<-log_GDP_ts
S=round(0.75*length(y))
h=2
error1.h_model1<-c()
error2.h_model1<-c()
for (i in S:(length(y)-h))
{
  mymodel.sub<-arima(y[1:i], order = c(1,1,1),seasonal=c(0,0,0))
  predict.h<-predict(mymodel.sub,n.ahead=h)$pred[h]
  error1.h_model1<-c(error1.h_model1,y[i+h]-predict.h)
  error2.h_model1<- c(error2.h_model1,(y[i+h]-predict.h)/y[i+h])
}
MAE_h1_1 <- mean(abs(error1.h_model1))
MAPE_h1_1 <- mean(abs(error2.h_model1))

error1.h_model2<-c()
error2.h_model2<-c()
for (i in S:(length(y)-h))
{
  mymodel.sub<-arima(y[1:i], order = c(1,1,2),seasonal=c(0,0,0))
  predict.h<-predict(mymodel.sub,n.ahead=h)$pred[h]
  error1.h_model2<-c(error1.h_model2,y[i+h]-predict.h)
  error2.h_model2<- c(error1.h_model2,(y[i+h]-predict.h)/y[i+h])
}

MAE_h1_2 <- mean(abs(error1.h_model2))
MAPE_h1_2 <- mean(abs(error2.h_model2))

library(forecast)
dm.test(error1.h_model1,error1.h_model2,h=h,power=1)
dm.test(error1.h_model1,error1.h_model2,h=h,power=2)




h=1
error1.h_model1<-c()
error2.h_model1<-c()
for (i in S:(length(y)-h))
{
  mymodel.sub<-arima(y[1:i], order = c(1,1,1),seasonal=c(0,0,0))
  predict.h<-predict(mymodel.sub,n.ahead=h)$pred[h]
  error1.h_model1<-c(error1.h_model1,y[i+h]-predict.h)
  error2.h_model1<- c(error2.h_model1,(y[i+h]-predict.h)/y[i+h])
}
MAE_h1_1 <- mean(abs(error1.h_model1))
MAPE_h1_1 <- mean(abs(error2.h_model1))

error1.h_model2<-c()
error2.h_model2<-c()
for (i in S:(length(y)-h))
{
  mymodel.sub<-arima(y[1:i], order = c(1,1,2),seasonal=c(0,0,0))
  predict.h<-predict(mymodel.sub,n.ahead=h)$pred[h]
  error1.h_model2<-c(error1.h_model2,y[i+h]-predict.h)
  error2.h_model2<- c(error1.h_model2,(y[i+h]-predict.h)/y[i+h])
}

MAE_h1_2 <- mean(abs(error1.h_model2))
MAPE_h1_2 <- mean(abs(error2.h_model2))

library(forecast)
dm.test(error1.h_model1,error1.h_model2,h=h,power=1)
dm.test(error1.h_model1,error1.h_model2,h=h,power=2)


uemp_ts <- ts (unemployment_AUSgium$unemployment_Value, frequency = 1, start = c(1970,1))
plot.ts(uemp_ts)

ts.plot(GDP_ts, uemp_ts, col=c("black","red"), main="Comparison of GDP and unemployment")


#log#
log_uemp_ts<- log (uemp_ts)
plot.ts(log_uemp_ts)
dloguemp_ts <- diff(log_uemp_ts)
plot.ts(dloguemp_ts, ylab="Change in unemployment", main="Change in unemployment")
CADFtest(uemp_ts, type ="drift", criterion="BIC", max.lag.y = 7) #p-value=0.20>0.05#I(1)
CADFtest(dloguemp_ts, type ="drift", criterion="BIC", max.lag.y = 7) #p-value=0<0.05#I(0)
CADFtest(log_GDP_ts, type ="drift", criterion="BIC", max.lag.y = 7)
CADFtest(dlogGDP_ts, type ="drift", criterion="BIC", max.lag.y = 7)#p-value is 0.16 not stationary

#####################BOTH SERIES ARE NOW STATIONARY#####################

ts.plot(dlogGDP_ts, dloguemp_ts, col=c("blue","red"), main="Comparison of GDP and unemployment")
#########################################################################################
sqrt(length(dlogGDP_ts))
sqrt(length(dloguemp_ts))
##
fit_ci_lm <-  lm(dlogGDP_ts ~ dloguemp_ts)
summary(fit_ci_lm)

# Test for Co-integration
fit_ci <-  lm(log_GDP_ts ~ log_uemp_ts)
summary(fit_ci)
res_fit_ci <- fit_ci$residuals
CADFtest(res_fit_ci,type="drift",criterion="BIC",max.lag.y=max.lag)
# We obtain a test statistics −0.62, which is larger that the Engle-Granger ADF test statistics for
# one explanatory variable −3.41. Thus, do not reject H0 of no co-integration and conclude that they are not co-integrated.
#pvalue>0.05, not co integrated!

plot.ts(fit_ci$residuals)
acf(fit_ci$residuals)
Box.test(fit_ci$residuals, lag = max.lag, type = "Ljung-Box")
##has p-value = 0.01 < 5%, thus we reject H0 and conclude that the model is not valid.

logdata<-data.frame(log(GDP_AUSgium$GDP_Value), log(unemployment_AUSgium$unemployment_Value))
dlogdata<-data.frame (diff(log(GDP_AUSgium$GDP_Value)), diff(log(unemployment_AUSgium$unemployment_Value)))
names(logdata)<-c("logGDP","logUEMP")
names(dlogdata)<-c("dlogGDP","dlogUEMP")
attach(logdata)

lag <- 4
n <- length(log_GDP_ts)
log_GDP.0 <- log_GDP_ts[(lag+1):n]
log_uemp.0 <- log_uemp_ts[(lag+1):n]
log_uemp.1 <- log_uemp_ts[lag:(n-1)]
log_uemp.2 <- log_uemp_ts[(lag-1):(n-2)]
log_uemp.3 <- log_uemp_ts[(lag-2):(n-3)]
log_uemp.4 <- log_uemp_ts[(lag-3):(n-4)]
fit_dlm2 <- lm(log_GDP.0 ~ log_uemp.0+log_uemp.1+log_uemp.2+log_uemp.3+log_uemp.4)
summary(fit_dlm2)
plot.ts(fit_dlm2$residuals)
acf(fit_dlm2$residuals)
Box.test(fit_dlm2$residuals, lag = max.lag, type = "Ljung-Box")
# reject the model p<0.05 

lag <- 3
log_GDP.0 <- log_GDP_ts[(lag+1):n]
log_uemp.0 <- log_uemp_ts[(lag+1):n]
log_uemp.1 <- log_uemp_ts[lag:(n-1)]
log_GDP.1 <- log_GDP_ts[lag:(n-1)]
log_GDP.2 <- log_GDP_ts[(lag-1):(n-2)]
log_uemp.2 <- log_uemp_ts[(lag-1):(n-2)]
log_GDP.3 <- log_GDP_ts[(lag-2):(n-3)]
log_uemp.3 <- log_uemp_ts[(lag-2):(n-3)]
fit_adlm <- lm(log_GDP.0 ~ log_GDP.1+log_GDP.2+log_GDP.3+log_uemp.1+log_uemp.2+log_uemp.3)
plot.ts(fit_adlm$residuals)
acf(fit_adlm$residuals)
Box.test(fit_adlm$residuals, lag = max.lag, type = "Ljung-Box")
summary(fit_adlm)
#Model is valid, overall p value is less than 0.5. but only variable log_GDP.1 is valid

##Granger causality using adlm(1)
lag <- 1
n <- length(dlogGDP_ts)
dlogGDP.0 <- dlogGDP_ts[(lag+1):n]
dlogGDP.1 <- dlogGDP_ts[lag:(n-1)]
dlogCS.1 <- dloguemp_ts[lag:(n-1)]
fit_dlm <- lm(dlogGDP.0 ~ dlogGDP.1+dlogCS.1)
acf(fit_dlm$residuals)
sqrt(length(dlogGDP_ts))
Box.test(fit_dlm$residuals, lag = max.lag, type = "Ljung-Box")
fit_dlm_nox <- lm(dlogGDP.0 ~ dlogGDP.1)
summary(fit_dlm_nox)
anova(fit_dlm,fit_dlm_nox)
#p value of 0.08, so we cannot reject H0 of no granger causality. 
#No granger causality

#GARCH 
spread<-ts(logGDP-logUEMP,frequency=1,start=c(1970,1))
ts.plot(spread)
max.lag<-round(sqrt((length(spread))))
CADFtest(spread,type="drift",criterion="BIC",max.lag.y=max.lag) #not stationary
#p-value = 0.9 > 5%, thus we cannot reject H0 and conclude that the spread is not stationary.

#GARCH 
spread<-ts(diff(logGDP)-diff(logUEMP),frequency=1,start=c(1970,1))
ts.plot(spread)
max.lag<-round(sqrt((length(spread))))
CADFtest(spread,type="drift",criterion="BIC",max.lag.y=max.lag) #stationary
#p-value = 0< 5%, thus we  reject H0 and conclude that the spread is stationary.

par(mfrow=c(1,1))
acf(spread)
pacf(spread) 
fit_ar<-arima(spread,order=c(1,0,1))
fit_ar
plot(fit_ar$residuals)
acf(fit_ar$residuals)
#no significant autocorrelations
Box.test(fit_ar$residuals,lag=max.lag,type="Ljung-Box") #white noise
#p-value = 0.97 > 5%, thus we do not reject H0 and conclude that the model is valid.
par(mfrow=c(1,1))
acf(fit_ar$residuals^2)
#There are some significant autocorrelations in the squared residuals. //there is 1
#We might be in presence of heteroskedasticity


## can also include just garch
library(fGarch)
fit_garch<-garchFit(~garch(1,1),data=spread, include.mean = F)
summary(fit_garch)
plot(fit_garch)



library(urca)
trace_test<- ca.jo(logdata, type="trace", K=2, ecdet="const", spec="transitory")
summary(trace_test)
#for r=0, the test statistic is larger than the critical value at all levels
#for r<=1, test statistic is larger than the critical value at 10 and 5 pct
#we reject the null hypothesis that
#the number of Cointegration Equations is equal to 0 at all levels, and cannot reject that there
#is 1 Cointegration Equation at 1 percent level.

eigen_test<- ca.jo(logdata, type="eigen",K=2, ecdet="const", spec="transitory")
summary(eigen_test)
##we reject the null hypothesis that
#the number of Cointegration Equations is equal to 0 at all levels,
#and cannot reject that there
#is 1 Cointegration Equation at 1 percent level.


fit_vecm<-cajorls(trace_test, r=1)
fit_vecm

library(vars)

VARselect(dlogdata,lag.max=7,type="const")
#lowest value SC=2
fit_varautom<-VAR(dlogdata,type="const",p=2)
summary(fit_varautom)

varautom_residuals<-resid(fit_varautom)


# win.graph()
par(mfrow=c(1,1))
acf(varautom_residuals[,1])
acf(varautom_residuals[,2])
ccf(varautom_residuals[,1],varautom_residuals[,2])
irf_var<-irf(fit_varautom,ortho=F,boot=T)
par(mar=c(1,1,1,1))
plot(irf_var)


fit_var<-vec2var(trace_test, r=1)
fit_var
my_forecast<-predict(fit_varautom, n.ahead=6)
my_forecast

forecasts <- predict(fit_varautom)
plot(forecasts)

