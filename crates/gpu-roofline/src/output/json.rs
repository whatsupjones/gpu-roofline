use crate::model::dynamic::DynamicRoofline;
use crate::model::roofline::RooflineModel;

/// Print a static roofline model as JSON.
pub fn print_static_json(model: &RooflineModel) {
    match serde_json::to_string_pretty(model) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Error serializing to JSON: {e}"),
    }
}

/// Print a dynamic roofline as JSON.
pub fn print_dynamic_json(dynamic: &DynamicRoofline) {
    match serde_json::to_string_pretty(dynamic) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Error serializing to JSON: {e}"),
    }
}
