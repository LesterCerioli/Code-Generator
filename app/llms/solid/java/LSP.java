public class LSP {


class Bird {
    void fly() {}
}


class Ostrich extends Bird {
    void fly() {
        throw new RuntimeException("Cannot fly");
    }
}


interface FlyingBird {
    void fly();
}

