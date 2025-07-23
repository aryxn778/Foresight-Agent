from pyspark.ml import PipelineModel

model = PipelineModel.load("model/demand_model")
print("Features column used by model:", model.stages[-1]._java_obj.getFeaturesCol())
print("Label column used by model:", model.stages[-1]._java_obj.getLabelCol())