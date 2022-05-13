import os
import unittest
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import FacilityBase, ConsumerAction, StorageUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import OneTimeSkuPriceDemandSampler, \
    DataFileDemandSampler


def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    env = Env(scenario="supply_chain", topology=case_folder, durations=durations)

    return env


def get_product_dict_from_storage(env: Env, frame_index: int, node_index: int):
    product_list = env.snapshot_list["storage"][frame_index:node_index:"product_list"].flatten().astype(np.int)
    product_quantity = env.snapshot_list["storage"][frame_index:node_index:"product_quantity"].flatten().astype(np.int)

    return {product_id: quantity for product_id, quantity in zip(product_list, product_quantity)}


SKU1_ID = 1
SKU2_ID = 2
SKU3_ID = 3
SKU4_ID = 4
FOOD_1_ID = 20
HOBBY_1_ID = 30


class MyTestCase(unittest.TestCase):
    """
        state only  test:
        . consumer_state_only
            . 'pending_order_daily'
        . seller_state_only
            . 'sale_mean'
            . 'sale_hist'
        . distribution_state_only
            . 'pending_order'
            . 'in_transit_orders'
    """

    def test_consumer_state_only(self) -> None:
        """Test the 'pending_order_daily' of the consumer unit."""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        distribution_unit = supplier_3.distribution
        consumer_unit = warehouse_1.products[3].consumer

        order = Order(warehouse_1, SKU3_ID, 10, "train")

        # There are 2 "train" in total, and 1 left after scheduling this order.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        warehouse_1_consumer_unit_id = 11

        # Here the vlt of "train" is less than "pending_order_daily" length
        self.assertEqual([0, 0, 0, 10], env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily'])
        self.assertEqual([0, 0, 0, 10], consumer_unit.pending_order_daily)

        # add another order, it would be successfully scheduled, but none available vehicle left now.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual([0, 0, 0, 10 + 10], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 0, 10 + 10], consumer_unit.pending_order_daily)

        start_tick = env.tick
        expected_tick = start_tick + 3  # vlt = 3

        # 3rd order, will cause the pending order increase
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        env.step(None)
        self.assertEqual([0, 0, 20, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 20, 0], list(consumer_unit.pending_order_daily))

        env.step(None)
        self.assertEqual([0, 20, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 20, 0, 0], list(consumer_unit.pending_order_daily))

        env.step(None)
        self.assertEqual([20, 0, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([20, 0, 0, 0], list(consumer_unit.pending_order_daily))

        env.step(None)
        self.assertEqual([0, 0, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 0, 0], list(consumer_unit.pending_order_daily))

        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_tick
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(10 * 1, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 2, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 1, distribution_unit.transportation_cost[SKU3_ID])
        self.assertEqual([0, 0, 10, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 10, 0], list(consumer_unit.pending_order_daily))

        self.assertEqual([0, 0, 10, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 10, 0], list(consumer_unit.pending_order_daily))

        distribution_unit.place_order(env.tick, order)

        self.assertEqual([0, 00, 10, 10], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 00, 10, 10], list(consumer_unit.pending_order_daily))

        start_tick = env.tick
        expected_tick = start_tick + 3 - 1  # vlt = 3
        while env.tick < expected_tick:
            env.step(None)

        self.assertEqual([10, 10, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([10, 10, 0, 0], list(consumer_unit.pending_order_daily))

    def test_seller_state_only(self) -> None:
        """Test "sale_mean" and "_sale_hist"""

        env = build_env("case_05", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        storeproductunit_sku1, storeproductunit_sku2, storeproductunit_sku3 = 1, 3, 2

        self.assertEqual([1, 1, 1, 1, 1, 1], Store_001.children[storeproductunit_sku1].seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 2], Store_001.children[storeproductunit_sku2].seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 3], Store_001.children[storeproductunit_sku3].seller._sale_hist)

        env.step(None)
        # The demand in the data file should be added after env.step, and now it is filled with 0 if it is not implemented.
        self.assertEqual([1, 1, 1, 1, 1, 10], Store_001.children[storeproductunit_sku1].seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 100], Store_001.children[storeproductunit_sku2].seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 100], Store_001.children[storeproductunit_sku3].seller._sale_hist)

        # The result should be (1+1+1+1+1+10)/6=0.8333333333333334
        self.assertEqual(2.5, env.metrics['products'][26]['sale_mean'])
        # The result should be (3+3+3+3+3+100)/6=19.166666666666668
        self.assertEqual(19.166666666666668, env.metrics['products'][29]['sale_mean'])
        # The result should be (2+2+2+2+2+100)/6=18.333333333333332
        self.assertEqual(18.333333333333332, env.metrics['products'][32]['sale_mean'])

    def test_distribution_state_only(self) -> None:
        """Test the 'pending_order' and 'in_transit_orders'of the distribution unit."""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        distribution_unit = supplier_3.distribution
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        Retailer_1: FacilityBase = be.world._get_facility_by_name("Retailer_001")
        consumer_unit = warehouse_1.products[3].consumer

        order = Order(warehouse_1, SKU3_ID, 20, "train")

        # There are 2 "train" in total, and 1 left after scheduling this order.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        warehouse_1_consumer_unit_id = 11

        # Here the vlt of "train" is less than "pending_order_daily" length
        self.assertEqual([0, 0, 0, 20], env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily'])
        self.assertEqual([0, 0, 0, 20], consumer_unit.pending_order_daily)

        # add another order, it would be successfully scheduled, but none available vehicle left now.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual([0, 0, 0, 20 + 20], env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily'])
        self.assertEqual([0, 0, 0, 20 + 20], consumer_unit.pending_order_daily)

        # 3rd order, will cause the pending order increase
        order_1 = Order(warehouse_1, SKU3_ID, 25, "train")
        distribution_unit.place_order(env.tick, order_1)
        supplier_3_id, warehouse_1_id, retailer_1_id = 1, 6, 13

        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(25, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(25, env.metrics['facilities'][supplier_3_id]['pending_order'][SKU3_ID])
        self.assertEqual(25, distribution_unit._pending_product_quantity[SKU3_ID])

        Warehouse_1_distribution_unit = warehouse_1.distribution
        retailer_1_consumer_unit = Retailer_1.products[3].consumer
        order_2 = Order(Retailer_1, SKU3_ID, 5, "train")

        Warehouse_1_distribution_unit.place_order(env.tick, order_2)
        Warehouse_1_distribution_unit.place_order(env.tick, order_2)
        self.assertEqual(0, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(0, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        # 3rd order, will cause the pending_order increase
        Warehouse_1_distribution_unit.place_order(env.tick, order_2)
        self.assertEqual(5, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(5, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        # There is no place_order for the distribution of supplier_3, there should be no change
        self.assertEqual(25, env.metrics['facilities'][supplier_3_id]['pending_order'][SKU3_ID])
        self.assertEqual(25, distribution_unit._pending_product_quantity[SKU3_ID])

        start_tick = env.tick
        expected_supplier_tick = start_tick + 3
        expected_warehouse_tick = start_tick + 7

        while env.tick < expected_supplier_tick - 1:
            env.step(None)

        self.assertEqual([40, 0, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([40, 0, 0, 0], list(consumer_unit.pending_order_daily))

        env.step(None)
        self.assertEqual([0, 0, 0, 0], list(env.metrics['products'][warehouse_1_consumer_unit_id]['pending_order_daily']))
        self.assertEqual([0, 0, 0, 0], list(consumer_unit.pending_order_daily))

        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_supplier_tick
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(25, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(5, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(5, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(-40, env.metrics['facilities'][warehouse_1_id]['in_transit_orders'][SKU3_ID])
        self.assertEqual(-40, consumer_unit._open_orders[supplier_3_id][SKU3_ID])

        self.assertEqual(5, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(5, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        while env.tick < expected_warehouse_tick:
            env.step(None)
        assert env.tick == expected_warehouse_tick

        # warehouse_1_distribution has a vlt of 7 and arrives 7+1 ticks later
        self.assertEqual(5, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(5, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        env.step(None)

        self.assertEqual(0, env.metrics['facilities'][warehouse_1_id]['pending_order'][SKU3_ID])
        self.assertEqual(0, Warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        # when the order arrives, Retailer_1's in_transit_orders shall be the negative of retailer_1's arrived orders -10
        self.assertEqual(-10, env.metrics['facilities'][retailer_1_id]['in_transit_orders'][SKU3_ID])
        self.assertEqual(-10, retailer_1_consumer_unit._open_orders[warehouse_1_id][SKU3_ID])

        # after the order arrives, the previous pending_order of supplier_3's distribution has also arrived, so warehouse_1's in_transit_orders should be -40 + (-25) equals -65.
        self.assertEqual(-65, env.metrics['facilities'][warehouse_1_id]['in_transit_orders'][SKU3_ID])
        self.assertEqual(-65, consumer_unit._open_orders[supplier_3_id][SKU3_ID])


if __name__ == '__main__':
    unittest.main()
