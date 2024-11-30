import sqlparse
from sqlparse.sql import Token, Identifier, Parenthesis
from lxml import etree


from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLearning,  # Changed from AutoModelForSeq2SeqGeneration
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import torch

def translate_to_openmath(schema):
    root = etree.Element("OMOBJ")
    oma = etree.SubElement(root, "OMA")
    etree.SubElement(oma, "OMS", cd="university", name="schema")
    
    for table in schema["tables"]:
        table_element = etree.SubElement(oma, "OMA")
        etree.SubElement(table_element, "OMS", cd="set1", name="set")
        etree.SubElement(table_element, "OMV", name=table["name"])
        
        for column in table["columns"]:
            column_element = etree.SubElement(table_element, "OMA")
            etree.SubElement(column_element, "OMS", cd="attribute", name=column["name"])
            etree.SubElement(column_element, "OMS", cd="datatype", name=column["type"])
    
    return etree.tostring(root, pretty_print=True).decode()

def parse_sql(sql):
    schema = {"tables": []}
    parsed = sqlparse.parse(sql)[0]  # Get first statement
    
    # Find table name and column definitions
    table = {"name": "", "columns": []}
    for token in parsed.tokens:
        if isinstance(token, Identifier) and token.get_parent_name() is None:
            table["name"] = token.value
        elif isinstance(token, Parenthesis):
            # Split column definitions
            col_defs = [c.strip() for c in token.value.strip('()').split(',')]
            for col_def in col_defs:
                if col_def:
                    # Parse each column definition
                    parts = col_def.strip().split()
                    col_name = parts[0]
                    col_type = parts[1]
                    is_primary = 'PRIMARY KEY' in col_def.upper()
                    
                    table["columns"].append({
                        "name": col_name,
                        "type": col_type,
                        "isPrimaryKey": is_primary
                    })
    
    schema["tables"].append(table)
    return schema

# Test the parser
sql_schema = """
CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    Name VARCHAR(100),
    Age INT,
    Major VARCHAR(100)
);
"""

parsed_schema = parse_sql(sql_schema)
print(translate_to_openmath(parsed_schema))




###############################

# Prepare data in correct dictionary format
def prepare_dataset(examples):
    data_dict = {
        "sql": [],
        "openmath": []
    }
    for example in examples:
        data_dict["sql"].append(example[0])
        data_dict["openmath"].append(example[1])
    return data_dict

# Example data
training_examples = [
    (sql_schema, translate_to_openmath(parsed_schema))
]

dataset_dict = prepare_dataset(training_examples)
dataset = Dataset.from_dict(dataset_dict)

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLearning.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["sql"],
        examples["openmath"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_data = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

trainer.train()

model.save_pretrained("./openmath-translator")
tokenizer.save_pretrained("./openmath-translator")