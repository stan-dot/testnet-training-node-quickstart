package main

import (
	"context"
	"fmt"
)

// App struct
type App struct {
	ctx context.Context
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

// Greet returns a greeting for the given name
func (a *App) Greet(name string) string {
	return fmt.Sprintf("Hello %s, It's show time!", name)
}


// fetchData fetches data from the FastAPI server
func (a *App) fetchData() (string, error) {
	// URL of the FastAPI endpoint
	url := "http://server:8000/data"

	// Make the HTTP GET request
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("error fetching data: %v", err)
	}
	defer resp.Body.Close()

	// Check for a successful response
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Read and parse the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %v", err)
	}

	// Unmarshal JSON (assuming the response is JSON)
	var data map[string]interface{}
	err = json.Unmarshal(body, &data)
	if err != nil {
		return "", fmt.Errorf("error parsing JSON: %v", err)
	}

	// Extracting data you want (adjust according to the FastAPI response structure)
	result, ok := data["result"].(string)
	if !ok {
		return "", fmt.Errorf("unexpected data format")
	}

	return result, nil
}
