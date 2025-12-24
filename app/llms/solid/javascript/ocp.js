
function calculateDiscount(type, price) {
  if (type === "student") return price * 0.9;
  if (type === "senior") return price * 0.8;
  return price;
}

class Discount {
  apply(price) {
    return price;
  }
}

class StudentDiscount extends Discount {
  apply(price) {
    return price * 0.9;
  }
}

class SeniorDiscount extends Discount {
  apply(price) {
    return price * 0.8;
  }
}
