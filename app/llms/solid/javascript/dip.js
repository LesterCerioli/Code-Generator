


class MySQLDatabase {
  connect() {}
}

class UserService {
  constructor() {
    this.database = new MySQLDatabase();
  }
}

class Database {
  connect() {}
}

class MySQLDatabase extends Database {
  connect() {}
}

class UserService {
  constructor(database) {
    this.database = database;
  }
}
