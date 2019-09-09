

/****** SSMS의 SelectTopNRows 명령 스크립트 ******/
SELECT TOP (1000) [SepalLength]
      ,[SepalWidth]
      ,[PetalLength]
      ,[PetalWidth]
      ,[Name]
  FROM [bitdb].[dbo].[iris2]