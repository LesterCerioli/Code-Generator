package main

type Report struct{}

func (r Report) Generate()   {}
func (r Report) SaveToFile() {}

type ReportGenerator struct{}

func (g ReportGenerator) Generate() {}

type ReportSaver struct{}

func (s ReportSaver) Save() {}
