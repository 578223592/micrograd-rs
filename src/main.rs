use micrograd_rs::{ Value};
use std::ops::Add;

fn main() {
    let node1 = Value::new(1.0);
    let node2 = Value::new(1.0);
    let node3 = node1.add(node2);
    println!("{:?}", node3);
    
    let node1 = Value::new(1.0);
    let node2 = 1.0;
    let node3 = node1.add(node2);
    println!("{:?}", node3);
    
    let node1 = Value::new(1.0);
    let node2 = 1.0;
    let node3 = node2 + node1;
    println!("{:?}", node3);
}
