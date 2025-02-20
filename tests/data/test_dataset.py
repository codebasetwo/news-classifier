import great_expectations as gx

def test_dataset(df, context):
    """Test dataset quality and integrity."""
        
    # Create an Expectation Suite
    suite_name = "nesfeed_expectation_suite"
    suite = gx.ExpectationSuite(name=suite_name)

    # Add the Expectation Suite to the Data Context
    suite = context.suites.add(suite)

    # Connect to data and create a Batch.
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Create expectaions
    column_set = ["ARTS, CULTURE & TRAVEL", "EDUCATION",
            "ENTERTAINMENT", "NEWS & POLITICS", "PARENTING", "SPORTS & WELLNESS"]

    column_name = list(df.columns)
    df.drop_duplicates(subset= ["headline", "short_description"], inplace=True, ignore_index=True)
    distinct_expectation = gx.expectations.ExpectColumnDistinctValuesToBeInSet(column="category", value_set=column_set) # expected labels
    compound_col_expectation = gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["headline", "short_description"]) # data leaks
    null_expectation = gx.expectations.ExpectColumnValuesToNotBeNull(column="category", mostly=0.9) # missing values
    headline_type_expectation = gx.expectations.ExpectColumnValuesToBeOfType(column="headline", type_="str") # type adherence
    description_type_expectation = gx.expectations.ExpectColumnValuesToBeOfType(column="short_description", type_="str") # type adherence
    available_col_expectation = gx.expectations.ExpectTableColumnsToMatchOrderedList(column_list=column_name) # schema adherence
    suite.add_expectation(distinct_expectation)
    suite.add_expectation(compound_col_expectation)
    suite.add_expectation(null_expectation)
    suite.add_expectation(headline_type_expectation)
    suite.add_expectation(description_type_expectation)
    suite.add_expectation(available_col_expectation)

    # Validate dataset
    validation_result = batch.validate(suite)
    assert validation_result["success"]