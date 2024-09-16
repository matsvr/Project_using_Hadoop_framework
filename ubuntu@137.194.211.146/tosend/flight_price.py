from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

spark = SparkSession.\
    builder.\
    appName("pyspark-notebook").\
    master("spark://192.168.3.142:7077").\
    config("spark.executor.memory", "512m").\
    getOrCreate()

df = spark.read.csv(path="extrait_flight.csv", sep=",",
                    header=True, inferSchema=True)
df = df.na.fill("UNKNOWN")

df.select("isBasicEconomy").show()

selected_columns = ["startingAirport", "destinationAirport", "fareBasisCode",
                    "travelDuration", "elapsedDays", "seatsRemaining", "totalTravelDistance",
                    "segmentsArrivalAirportCode", "segmentsDepartureAirportCode", "segmentsAirlineCode",
                    "segmentsEquipmentDescription", "segmentsDurationInSeconds", "segmentsDistance", "segmentsCabinCode",
                    "totalFare"]

# Indexer les colonnes de chaînes de caractères
string_columns = ["startingAirport", "destinationAirport", "fareBasisCode",
                  "travelDuration",
                  "segmentsArrivalAirportCode", "segmentsDepartureAirportCode", "segmentsAirlineCode",
                  "segmentsEquipmentDescription", "segmentsDurationInSeconds", "segmentsDistance", "segmentsCabinCode"]

string_indexers = [StringIndexer(
    inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in string_columns]

# Gérer les caractéristiques catégorielles
# categorical_columns = ["isBasicEconomy", "isRefundable", "isNonStop", "seatsRemaining"]
# selected_columns2 = list(filter(lambda x: x not in string_columns, selected_columns))
# selected_columns2 = list(filter(lambda x: x not in categorical_columns, selected_columns2))

# categorical_indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_columns]
# categorical_encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_columns]

# Assembler les colonnes en une seule colonne "features"
feature_columns = [
    col+"_index" if col in string_columns else col for col in selected_columns[:-1]]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

print(feature_columns)

rf = RandomForestRegressor(featuresCol="features",
                           labelCol="totalFare", maxBins=200000)

pipeline = Pipeline(stages=string_indexers + [assembler, rf])

(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(training_data)

predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol="totalFare", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")

spark.stop()
