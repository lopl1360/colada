from trading_app.models import Order


def save_order_if_valid(order_obj, side, fallback_price=0.0):
    """
    Saves the order to the DB if it's a valid Alpaca order object.
    """
    if hasattr(order_obj, "symbol"):
        db_order = Order(
            symbol=order_obj.symbol,
            side=side,
            qty=order_obj.qty,
            type=order_obj.type,
            status=order_obj.status,
            price=float(getattr(order_obj, "filled_avg_price", None) or fallback_price),
        )
        db_order.save()
