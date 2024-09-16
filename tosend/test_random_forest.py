from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import to_date, dayofweek, dayofmonth, month, year
from pyspark.sql import functions as F

# Initialisation de la session Spark
spark = SparkSession.builder.appName("FlightPricePrediction").getOrCreate()

# Chargement des données
df = spark.read.csv('extrait_flight.csv', header=True, inferSchema=True)

# Sélection des 15 premières colonnes et de la colonne cible
selected_columns = df.columns[:15]
df_selected = df.select(selected_columns)

# Liste des colonnes de date, des colonnes numériques et des colonnes booléennes
bool_columns = ["isBasicEconomy", "isRefundable", "isNonStop"]
date_columns = ["searchDate", "flightDate"]
numeric_columns = ["elapsedDays", "baseFare",
                   "totalFare", "seatsRemaining", "totalTravelDistance"]
string_columns = ["legId", "startingAirport",
                  "destinationAirport", "fareBasisCode", "travelDuration"]

# Conversion des colonnes booléennes en entiers
for bool_col in bool_columns:
    df = df.withColumn(
        bool_col, F.col(bool_col).cast("integer"))

# Conversion des chaînes de caractères de date en type de date
for date_col in date_columns:
    df = df.withColumn(
        date_col, to_date(df[date_col], 'yyyy-MM-dd'))

# Extraction des composantes de la date
for date_col in date_columns:
    df = df.withColumn(date_col + "_year", year(F.col(date_col)))\
        .withColumn(date_col + "_month", month(F.col(date_col)))\
        .withColumn(date_col + "_dayOfMonth", dayofmonth(F.col(date_col)))\
        .withColumn(date_col + "_dayOfWeek", dayofweek(F.col(date_col)))
    
# Création de la liste des colonnes pour l'assembleur de vecteurs
date_features = [col + "_" + feature for col in date_columns for feature in [
    "year", "month", "dayOfMonth", "dayOfWeek"]]
    
# Indexation des colonnes catégorielles
categorical_columns = [field for (field, dataType) in df_selected.dtypes if dataType == "string"]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_indexed").fit(df_selected) for column in categorical_columns]

# Assemblage des vecteurs de caractéristiques
assembler_inputs = [c + "_indexed" for c in categorical_columns] + [c for c in bool_columns if c not in categorical_columns] + date_features + numeric_columns

df_columns = df.columns
for col in assembler_inputs:
    assert col in df_columns, f"La colonne '{col}' n'existe pas dans le DataFrame."

vectorAssembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Création du modèle Random Forest
rf = RandomForestRegressor(labelCol="totalFare", featuresCol="features")

# Construction du pipeline
pipeline = Pipeline(stages=indexers + [vectorAssembler, rf])

# Division des données en ensembles d'entraînement et de test
train_data, test_data = df_selected.randomSplit([0.8, 0.2], seed=42)

# Entraînement du modèle
model = pipeline.fit(train_data)

# Prédiction sur l'ensemble de test
predictions = model.transform(test_data)

# Évaluation du modèle
evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) sur l'ensemble de test = {rmse}")

# Fermeture de la session Spark
spark.stop()
