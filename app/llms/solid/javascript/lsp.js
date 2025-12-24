

class Bird {
  fly() {
    console.log("Flying");
  }
}


class Ostrich extends Bird {
  fly() {
    throw new Error("Ostrich cannot fly");
  }
}

class FlyingBird {
  fly() {}
}

class Sparrow extends FlyingBird {
  fly() {
    console.log("Flying");
  }
}

class Ostrich {}
