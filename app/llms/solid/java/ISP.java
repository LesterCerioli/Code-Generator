
interface Worker {
    void work();
    void eat();
}

class Robot implements Worker {
    public void work() {}
    public void eat() {}
}


interface Workable {
    void work();
}

interface Eatable {
    void eat();
}
