        
        ##MEMORY REQUIREMENTS
        ##print(paste(c(round(2075259*9*8/(2^20), 2)), " MB"))
        ##(2075259 rows)*(9 columns)*(8 bytes/numeric)/(2^20 bytes/MB)
        ##142.5 MB
        
        ##READ DATA and SUBSET 2/1/2007 - 2/2/2007
        hpc <- read.table("./household_power_consumption.txt", 
                          header = TRUE, sep = ";")
        hpc <- subset(hpc, Date == "1/2/2007" | Date == "2/2/2007")
        
        ##CONVERT Date and Time variables to Date and POSIX classes
        ##factor to numeric and KEEPS VALUE
        hpc$Time <- strptime(paste(hpc$Date, hpc$Time, sep = " "), 
                             format = "%d/%m/%Y %H:%M:%S")
        hpc$Date <- as.Date(hpc$Date, format = "%d/%m/%Y")
        
        hpc$Voltage <- as.numeric(levels(hpc$Voltage)[hpc$Voltage]) ##Keeps value!!
        hpc$Global_active_power <- as.numeric(levels(hpc$Global_active_power)
                                              [hpc$Global_active_power])
        hpc$Global_reactive_power <- as.numeric(levels(hpc$Global_reactive_power)
                                                [hpc$Global_reactive_power])
        
        
        ##PLOT 1
        ########
        png(filename = "plot1.png", width = 480, height = 480)
        hist(hpc$Global_active_power, breaks = 16, col = "red", 
             xlab = "Global Active Power (kilowatts)", 
             main = "Global Active Power")
        dev.off()
        
        ##PLOT 2
        ########
        png(filename = "plot2.png", width = 480, height = 480)
        plot(hpc$Time, hpc$Global_active_power, type = "l",
             xlab = "",
             ylab = "Global Active Power (kilowatts)")
        dev.off()
        
        ##PLOT 3
        ########
        png(filename = "plot3.png", width = 480, height = 480)
        plot(hpc$Time, 
             as.numeric(levels(hpc$Sub_metering_1)[hpc$Sub_metering_1]), 
             type = "l",
             xlab = "",
             ylab = "Energy Sub Metering")
        points(hpc$Time, 
               as.numeric(levels(hpc$Sub_metering_2)[hpc$Sub_metering_2]), 
               type = "l", 
               col = "red")
        points(hpc$Time, hpc$Sub_metering_3, type = "l", col = "blue")
        legend("topright", 
               c("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"), 
               col = c("black", "red", "blue"), 
               lty = c(1, 1))
        dev.off()
        
        ##PLOT 4
        ########
        png(filename = "plot4.png", width = 480, height = 480)
        par(mfrow = c(2,2))
        
        plot(hpc$Time, hpc$Global_active_power, type = "l",
             xlab = "",
             ylab = "Global Active Power")
        
        plot(hpc$Time, hpc$Voltage, type = "l",
             xlab = "datetime",
             ylab = "Voltage")
        
        plot(hpc$Time, 
             as.numeric(levels(hpc$Sub_metering_1)[hpc$Sub_metering_1]), 
             type = "l",
             xlab = "",
             ylab = "Energy Sub Metering")
        points(hpc$Time, 
               as.numeric(levels(hpc$Sub_metering_2)[hpc$Sub_metering_2]), 
               type = "l", 
               col = "red")
        points(hpc$Time, hpc$Sub_metering_3, type = "l", col = "blue")
        legend("topright", 
               c("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"), 
               col = c("black", "red", "blue"), 
               lty = c(1, 1),
               bty = "n",
               xjust = 1,
               yjust = 0,
               cex = .75)
        
        plot(hpc$Time, hpc$Global_reactive_power, type = "l",
             xlab = "datetime",
             ylab = "Global Reactive Power")
        
        dev.off()
        
        #########