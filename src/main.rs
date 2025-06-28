use std::ops::{Div, Mul};
use micrograd_rs::Value;

fn main() {

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(3.0, "node2".to_string());
    let node3 = &node1 * &node2;


    node3.backward();

    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // ValueInner { data: 3.0, grad: 1.0, _op: "*", name: "" }
    // ValueInner { data: 3.0, grad: 1.0, _op: "", name: "node2" }
    // ValueInner { data: 1.0, grad: 3.0, _op: "", name: "node1" }



    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(1.0, "node2".to_string());
    let mut node3 = &node1 + &node2;
    

    node3.backward();

    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // Value { data: 1.0, grad: 1.0, _op: "", name: "node1" }
    // Value { data: 1.0, grad: 1.0, _op: "", name: "node2" }
    // Value { data: 2.0, grad: 1.0, _op: "+", name: "" }
    
    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 =Value::new_with_name(3.0, "node2".to_string());

    node3 = node1.pow_i(&node2);
    node3.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // Value { data: 1.0, grad: 3.0, _op: "", name: "node1" }
    // Value { data: 3.0, grad: 0.0, _op: "", name: "node2" }
    // Value { data: 1.0, grad: 1.0, _op: "pow", name: "" }


    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 =Value::new_with_name(3.0, "node2".to_string());
    let node3 =Value::new_with_name(-1.0, "node3".to_string());
    
    let node4 = node2.pow_i(&node3);
    let node5 = node1.mul(&node4);
    node5.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);

    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 =Value::new_with_name(3.0, "node2".to_string());
    
    let node3 = node1.div(&node2);
    node3.backward();
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
    // =
    
    
    // let value2 =3.0+ &node1 ;  要实现数字在左边，要么就用宏快速的为各种整数类型来实现，要么就只能数字在右边
}
