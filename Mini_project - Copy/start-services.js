#!/usr/bin/env node

/**
 * WaveFarer Service Manager
 * =========================
 * 
 * This script manages both Flask and Express services
 * for the WaveFarer microservices architecture.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { config, validateConfig } = require('./config');

// =============================================================================
// CONFIGURATION
// =============================================================================

const SERVICES = {
  flask: {
    name: 'Flask API (ML Services)',
    command: 'python',
    args: ['app.py'],
    cwd: process.cwd(),
    port: config.apis.flask.port,
    color: '\x1b[36m' // Cyan
  },
  express: {
    name: 'Express API (User Services)',
    command: 'node',
    args: ['index.js'],
    cwd: path.join(process.cwd(), 'backend'),
    port: config.apis.express.port,
    color: '\x1b[35m' // Magenta
  }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function log(service, message) {
  const timestamp = new Date().toISOString();
  const color = SERVICES[service].color;
  const reset = '\x1b[0m';
  console.log(`${color}[${timestamp}] ${SERVICES[service].name}:${reset} ${message}`);
}

function logError(service, message) {
  const timestamp = new Date().toISOString();
  const color = '\x1b[31m'; // Red
  const reset = '\x1b[0m';
  console.error(`${color}[${timestamp}] ERROR ${SERVICES[service].name}:${reset} ${message}`);
}

function logInfo(message) {
  const timestamp = new Date().toISOString();
  const color = '\x1b[32m'; // Green
  const reset = '\x1b[0m';
  console.log(`${color}[${timestamp}] INFO:${reset} ${message}`);
}

// =============================================================================
// SERVICE MANAGEMENT
// =============================================================================

class ServiceManager {
  constructor() {
    this.processes = new Map();
    this.isShuttingDown = false;
  }

  /**
   * Start a service
   */
  startService(serviceName) {
    const service = SERVICES[serviceName];
    if (!service) {
      throw new Error(`Unknown service: ${serviceName}`);
    }

    logInfo(`Starting ${service.name}...`);

    // Check if service directory exists
    if (!fs.existsSync(service.cwd)) {
      logError(serviceName, `Directory not found: ${service.cwd}`);
      return false;
    }

    // Start the process
    const process = spawn(service.command, service.args, {
      cwd: service.cwd,
      stdio: 'pipe',
      env: { ...process.env, NODE_ENV: config.environment }
    });

    // Store process reference
    this.processes.set(serviceName, process);

    // Handle stdout
    process.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        log(serviceName, output);
      }
    });

    // Handle stderr
    process.stderr.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        logError(serviceName, output);
      }
    });

    // Handle process exit
    process.on('exit', (code, signal) => {
      if (!this.isShuttingDown) {
        logError(serviceName, `Process exited with code ${code} (signal: ${signal})`);
        logInfo(`Restarting ${service.name} in 5 seconds...`);
        setTimeout(() => this.startService(serviceName), 5000);
      }
    });

    // Handle process errors
    process.on('error', (error) => {
      logError(serviceName, `Failed to start: ${error.message}`);
    });

    logInfo(`${service.name} started successfully`);
    return true;
  }

  /**
   * Stop a service
   */
  stopService(serviceName) {
    const process = this.processes.get(serviceName);
    if (process) {
      logInfo(`Stopping ${SERVICES[serviceName].name}...`);
      process.kill('SIGTERM');
      this.processes.delete(serviceName);
    }
  }

  /**
   * Stop all services
   */
  stopAllServices() {
    this.isShuttingDown = true;
    logInfo('Shutting down all services...');
    
    for (const [serviceName, process] of this.processes) {
      logInfo(`Stopping ${SERVICES[serviceName].name}...`);
      process.kill('SIGTERM');
    }
    
    this.processes.clear();
  }

  /**
   * Start all services
   */
  startAllServices() {
    logInfo('Starting WaveFarer microservices...');
    
    // Validate configuration
    const errors = validateConfig();
    if (errors.length > 0) {
      logError('config', 'Configuration errors found:');
      errors.forEach(error => logError('config', `- ${error}`));
      process.exit(1);
    }

    // Start Flask API
    this.startService('flask');

    // Start Express API after a short delay
    setTimeout(() => {
      this.startService('express');
    }, 2000);

    logInfo('All services started!');
    logInfo(`Flask API: http://localhost:${config.apis.flask.port}`);
    logInfo(`Express API: http://localhost:${config.apis.express.port}`);
  }

  /**
   * Get service status
   */
  getStatus() {
    const status = {};
    for (const [serviceName, process] of this.processes) {
      status[serviceName] = {
        name: SERVICES[serviceName].name,
        pid: process.pid,
        alive: !process.killed
      };
    }
    return status;
  }
}

// =============================================================================
// COMMAND LINE INTERFACE
// =============================================================================

function showHelp() {
  console.log(`
WaveFarer Service Manager
=========================

Usage: node start-services.js [command]

Commands:
  start     Start all services
  stop      Stop all services
  restart   Restart all services
  status    Show service status
  help      Show this help message

Examples:
  node start-services.js start
  node start-services.js status
  node start-services.js stop
`);
}

// =============================================================================
// MAIN EXECUTION
// =============================================================================

const manager = new ServiceManager();
const command = process.argv[2] || 'start';

// Handle graceful shutdown
process.on('SIGINT', () => {
  logInfo('Received SIGINT, shutting down gracefully...');
  manager.stopAllServices();
  process.exit(0);
});

process.on('SIGTERM', () => {
  logInfo('Received SIGTERM, shutting down gracefully...');
  manager.stopAllServices();
  process.exit(0);
});

// Execute command
switch (command) {
  case 'start':
    manager.startAllServices();
    break;
    
  case 'stop':
    manager.stopAllServices();
    break;
    
  case 'restart':
    manager.stopAllServices();
    setTimeout(() => manager.startAllServices(), 2000);
    break;
    
  case 'status':
    const status = manager.getStatus();
    console.log('\nService Status:');
    console.log('===============');
    for (const [serviceName, info] of Object.entries(status)) {
      const statusIcon = info.alive ? '🟢' : '🔴';
      console.log(`${statusIcon} ${info.name} (PID: ${info.pid})`);
    }
    break;
    
  case 'help':
    showHelp();
    break;
    
  default:
    console.error(`Unknown command: ${command}`);
    showHelp();
    process.exit(1);
} 