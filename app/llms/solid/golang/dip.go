package main

type MySQLDatabase struct{}

func (db MySQLDatabase) Connect() {}

type UserService struct {
	db MySQLDatabase
}

type Database interface {
	Connect()
}

type MySQL struct{}

func (m MySQL) Connect() {}

type UserService struct {
	db Database
}
