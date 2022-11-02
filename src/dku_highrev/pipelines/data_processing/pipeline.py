"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_orders, prepare_customers, label_customers

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
           node(
                func=prepare_orders,
                inputs="orders",
                outputs="orders_by_customers",
                name="prepare_orders_node",
            ),
            node(
                func=prepare_customers,
                inputs=["customers", "orders_by_customers"],
                outputs="customers_prepared",
                name="prepare_customers_node",
            ),
            node(
                func=label_customers,
                inputs="customers_prepared",
                outputs="customers_labeled",
                name="label_customers_node",
            ),
            
        ]
    )
