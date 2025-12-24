
class DiscountService {
    double apply(String type, double price) {
        if (type.equals("student"))
            return price * 0.9;
        return price;
    }
}

interface Discount {
    double apply(double price);
}

class StudentDiscount implements Discount {
    public double apply(double price) {
        return price * 0.9;
    }
}
