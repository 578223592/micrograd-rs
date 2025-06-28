use micrograd_rs::Value;

fn main() {
    let node1 = Value::new_with_name(1.0, "node1".to_string());
    let node2 = Value::new_with_name(1.0, "node2".to_string());
    let node3 = &node1 + &node2;
    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);

    node3.backward();

    println!("{:?}", node1);
    println!("{:?}", node2);
    println!("{:?}", node3);
}
