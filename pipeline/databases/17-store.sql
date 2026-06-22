-- Creates a trigger that decreases item quantity after an order.
DROP TRIGGER IF EXISTS after_order;

DELIMITER //

CREATE TRIGGER after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
END//

DELIMITER ;
