from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, dayofweek, dayofmonth, month, year
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

# Créer une session Spark
spark = SparkSession.\
    builder.\
    appName("pyspark-notebook").\
    master("spark://spark-master:7077").\
    config("spark.executor.memory", "512m").\
    getOrCreate()

# Lecture du fichier CSV
flightprices = spark.read.csv(
    "./extrait_flight.csv", header=True, inferSchema=True)

# Sélection des 15 premières colonnes
first15_columns = flightprices.columns[:15]

# Sélectionner uniquement ces colonnes
flightprices = flightprices.select(*first15_columns)

# Afficher df
flightprices.show(n=10)

# Noms des colonnes à supprimer
# colonnes_a_supprimer = ["segmentsDepartureTimeRaw", "segmentsArrivalTimeRaw", "segmentsDepartureTimeEpochSeconds", "segmentsArrivalTimeEpochSeconds"]

# Supprimer les colonnes indésirables
# df = df.drop(*colonnes_a_supprimer)

# Liste des colonnes de date, colonnes numériques et booléennes
# bool_columns = ["isBasicEconomy", "isRefundable", "isNonStop"]
# date_columns = ["searchDate", "flightDate"]
# numeric_columns = ["elapsedDays", "baseFare", "totalFare", "seatsRemaining", "totalTravelDistance", "segmentsDurationInSeconds", "segmentsDistance"]

# Liste des colonnes de date, des colonnes numériques et des colonnes booléennes
bool_columns = ["isBasicEconomy", "isRefundable", "isNonStop"]
date_columns = ["searchDate", "flightDate"]
numeric_columns = ["elapsedDays", "baseFare",
                   "totalFare", "seatsRemaining", "totalTravelDistance"]
string_columns = ["legId", "startingAirport",
                  "destinationAirport", "fareBasisCode", "travelDuration"]

# Conversion des colonnes booléennes en entiers
for bool_col in bool_columns:
    flightprices = flightprices.withColumn(
        bool_col, F.col(bool_col).cast("integer"))

# Conversion des chaînes de caractères de date en type de date
for date_col in date_columns:
    flightprices = flightprices.withColumn(
        date_col, to_date(flightprices[date_col], 'yyyy-MM-dd'))

# Extraction des composantes de la date
for date_col in date_columns:
    flightprices = flightprices.withColumn(date_col + "_year", year(date_col))\
        .withColumn(date_col + "_month", month(date_col))\
        .withColumn(date_col + "_dayOfMonth", dayofmonth(date_col))\
        .withColumn(date_col + "_dayOfWeek", dayofweek(date_col))

# Encodage des colonnes catégorielles
# string_columns = [col for col, dtype in df.dtypes if dtype == 'string']
indexers = [StringIndexer(
    inputCol=c, outputCol="{0}_indexed".format(c)) for c in string_columns]
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(
), outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]

# Création de la liste des colonnes pour l'assembleur de vecteurs
date_features = [col + "_" + feature for col in date_columns for feature in [
    "year", "month", "dayOfMonth", "dayOfWeek"]]
assembler_inputs = [encoder.getOutputCol()
                    for encoder in encoders] + date_features + numeric_columns

# Assemblage des vecteurs de fonctionnalités
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Modèle de régression linéaire
lr = LinearRegression(featuresCol="features", labelCol="totalFare")

# Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])

# Division des données
(train_data, test_data) = flightprices.randomSplit([0.8, 0.2])

# Entraînement du modèle
model = pipeline.fit(train_data)

# Prédictions
predictions = model.transform(test_data)

# Évaluation
evaluator = RegressionEvaluator(
    labelCol="totalFare", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
