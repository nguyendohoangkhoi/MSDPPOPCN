FOR /f "delims=" %%i IN ('DIR *.xlsx /b') DO ExcelToCSV.vbs "%%i" "%%i.csv"
