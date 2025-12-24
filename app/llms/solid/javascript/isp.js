
class Worker {
  work() {}
  eat() {}
}

class Robot extends Worker {
  eat() {
    throw new Error("Robot does not eat");
  }
}


class Workable {
  work() {}
}

class Eatable {
  eat() {}
}

class HumanWorker extends Workable {
  eat() {}
}

class RobotWorker extends Workable {}
