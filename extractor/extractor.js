var system = require('system');
var fs     = require('fs');
var page   = require('webpage').create();
var json   = require('./lib/json2');

// set viewport size
page.viewportSize = {
    width   : 1600,
    height  : 4000
};

// debug info
page.onResourceRequested = function(request) {
    system.stderr.write(JSON.stringify(request, undefined, 2));
    system.stderr.write('\n\n');
};

// debug info
page.onResourceReceived = function(request) {
    system.stderr.write(JSON.stringify(request, undefined, 2));
    system.stderr.write('\n\n');
};

// debug info
page.onConsoleMessage = function(message) {
    system.stderr.write(message);
    system.stderr.write('\n\n');
};

// pretty print
var pretty = function(object) {
    console.log(JSON.stringify(object, undefined, 2));
    console.log('\n\n');
};

// handle page load
page.onLoadFinished = function(status) {
    // unable to load page?
    if(status !== 'success') {
        return phantom.exit();
    }

    var data        = {};

    // put up our helpers
    page.evaluate(function() {
        // add trim functionality
        String.prototype.trim = function(string) {
            return string.replace(/^\s+|\s+$/g, '');
        };

        // well nodelist doesn't have it's own iterator
        NodeList.prototype.forEach = Array.prototype.forEach;

        // this thing has also no iterator
        CSSStyleDeclaration.prototype.forEach = Array.prototype.forEach;

        // this thing has also no iterator
        DOMTokenList.prototype.forEach = Array.prototype.forEach;

        // global utility functions
        window._utils = {
            // trim string
            trim    : function(string) {
                return (string) ? string.replace(/^\s+|\s+$/g, '') : '';
            },

            // clean string
            clean : function(string) {
                return (string) ? string.replace(/\r?\n|\r/g, '') : '';
            },

            // check if valid
            isValid : function(value) {
                var re = /^[a-zA-Z][a-zA-Z0-9\-_]+$/;

                return value && re.test(value);
            },

            // get element bound
            bound   : function(element) {
                // get scroll top
                var scrollTop  = document.documentElement.scrollTop  || document.body.scrollTop;
                // get scroll left
                var scrollLeft = document.documentElement.scrollLeft || document.body.scrollLeft;

                // get bounding client rectangle
                var rect = element.getBoundingClientRect();

                return {
                    width   : rect.width,
                    height  : rect.height,
                    left    : rect.left + scrollLeft,
                    top     : rect.top + scrollTop
                };
            },

            // get element description
            element : function(element, nameOnly) {
                var name = element.tagName.toLowerCase();
                var self = this;

                if(nameOnly) {
                    return name;
                }

                var classes = [];

                element.classList.forEach(function(key) {
                    if(self.isValid(key)) {
                        classes.push(key);
                    }
                });

                return {
                    name    : name,
                    id      : this.isValid(element.id) ? element.id : '',
                    classes : classes.sort()
                };
            },

            // generate tag path
            path : function(element, nameOnly) {
                var path = [];

                while(element) {
                    if(element === document.body) {
                        break;
                    }

                    path.splice(0, 0, this.element(element, nameOnly))

                    element = element.parentElement;
                }

                path.sort()

                return path;
            },

            // calculate computed css
            computed : function(element) {
                // get default computed style
                var defaults = document.defaultView.getComputedStyle(document.body);
                // get the computed style for target element
                var computed = document.defaultView.getComputedStyle(element);

                var data = {};

                computed.forEach(function(key) {
                    // don't care about dimension, let bound track that
                    if(['width', 'height', 'top', 'left', 'right', 'bottom'].indexOf(key) !== -1) {
                        return;
                    }

                    // don't care about webkit specific
                    if(key.charAt(0) === '-') {
                        return;
                    }

                    // don't care about default value
                    if(computed[key] === defaults[key]) {
                        return;
                    }

                    data[key] = defaults[key];
                });

                return data;
            }
        };
    });

    // extract meta data
    data = page.evaluate(function() {
        var titles          = [];
        var descriptions    = [];
        var gold_text       = [];

        // iterate on each titles
        document
        .querySelectorAll('title')
        .forEach(function(el) {
            titles.push(_utils.trim(el.innerText));
            gold_text.push(_utils.trim(el.innerText))
        });

        // iterate on each meta descriptions
        document
        .querySelectorAll('meta[name="description"]')
        .forEach(function(el) {
            descriptions.push(el.content);
            gold_text.push(el.content);
        });

        // open graph title
        document
        .querySelectorAll('meta[name="og:title"], meta[property="og:title"]')
        .forEach(function(el) {
            titles.push(_utils.trim(el.content));
            gold_text.push(_utils.trim(el.content));
        });

        // open graph description
        document
        .querySelectorAll('meta[name="og:description"], meta[property="og:description"]')
        .forEach(function(el) {
            descriptions.push(el.content);
            gold_text.push(el.content);
        });

        // twitter title
        document
        .querySelectorAll('meta[name="twitter:title"], meta[property="twitter:title"]')
        .forEach(function(el) {
            titles.push(_utils.trim(el.content));
            gold_text.push(_utils.trim(el.content));
        });

        // twitter description
        document
        .querySelectorAll('meta[name="twitter:description"], meta[property="twitter:description"]')
        .forEach(function(el) {
            descriptions.push(el.content);
            gold_text.push(_utils.trim(el.content));
        });

        return {
            url             : window.location.href,
            hostname        : window.location.hostname,
            titles          : titles,
            descriptions    : descriptions,
            gold_text       : gold_text
        }
    });

    // extract body
    data.body = page.evaluate(function() {
        // computed style for body
        var computed = {};

        // iterate on each computed styles
        document
        .defaultView
        .getComputedStyle(document.body)
        .forEach(function(key) {
            // don't care about webkit specific
            if(key.charAt(0) === '-') {
                return;
            }

            // don't care about default value
            computed[key] = document.defaultView.getComputedStyle(document.body)[key];
        });

        return {
            scroll : {
                top  : document.documentElement.scrollTop  || document.body.scrollTop,
                left : document.documentElement.scrollLeft || document.body.scrollLeft
            },
            bound       : _utils.bound(document.body),
            computed    : computed
        }
    });

    // extract links
    data.links = page.evaluate(function() {
        var links = [];

        document
        .querySelectorAll('a[href]')
        .forEach(function(el) {
            links.push(el.href);
        });

        return links;
    });

    // extract texts
    data.texts = page.evaluate(function() {
        var texts = [];

        // walk over all text in the page
        var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);

        // iterate on each node
        while(text = walker.nextNode()) {
            // no text?
            if(_utils.trim(text.nodeValue).length === 0) {
                continue;
            }

            // container node
            var node  = text.parentElement
            // get element bound
            var bound = _utils.bound(node);
            // get element name
            var name  = _utils.element(node, true);

            // skip styles and scripts
            if(['style', 'script', 'noscript'].indexOf(name) !== -1) {
                continue;
            }

            // no bound? skip it ..
            if((bound.width * bound.height) < 0) {
                continue;
            }

            // find the parent node that is a block
            while(node) {
                // get computed styles
                var computed = document.defaultView.getComputedStyle(node);

                // do we find the block parent?
                if((parseInt(computed.width) * parseInt(computed.height)) > 0) {
                    break;
                }

                // get parent element
                node = node.parentElement;

                // still a valid node?
                if(!node) {
                    break;
                }
            }

            // have we seen this node?
            if(node.__features) {
                // just push the text
                node.__features.text.push(_utils.clean(_utils.trim(text.nodeValue)));
                continue;
            }

            // collect features
            node.__features = {
                element  : _utils.element(node),
                path     : _utils.path(node, true),
                selector : _utils.path(node),
                text     : [_utils.clean(_utils.trim(text.nodeValue))],
                html     : node.innerHTML,
                bound    : _utils.bound(node),
                computed : _utils.computed(node)
            }

            // push text features
            texts.push(node.__features);

            // debug
            node.style.border = '1px solid red';
        }       

        return texts;
    });

    // extract images
    data.images = page.evaluate(function() {
        var images = [];

        // iterate on all images
        document
        .querySelectorAll('img[src]')
        .forEach(function(el) {
            var bound = _utils.bound(el);

            // has a valid bound?
            if((bound.width * bound.height) === 0) {
                return;
            }

            // push images
            images.push({
                src      : el.src,
                element  : _utils.element(el),
                path     : _utils.path(el, true),
                selector : _utils.path(el),
                bound    : bound,
                computed : _utils.computed(el)
            });

            // debug
            el.style.border = '1px solid red';
        })

        return images;
    });

    // set the output path
    var output = system.args[2] + '.json';
    // set the render path
    var render = system.args[2] + '.png';
    
    // save output
    fs.write(output, JSON.stringify(data, undefined, 4));
    // render page
    // page.render(render);

    return phantom.exit();
};

// invalid arguments?
if(system.args.length !== 3) {
    console.log('usage: phantomjs ' + system.args[0] + ' <url> <label>');
    phantom.exit();
}

// load page
page.open(system.args[1]);